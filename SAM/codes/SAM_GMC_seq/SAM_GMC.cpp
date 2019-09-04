#include <iostream>
#include <iomanip>
#include <omp.h>
#include "SAM_GMC.hpp"
#include "../GMC.hpp"
#include "../MyMath.hpp"

// SAM_GMC
SAM_GMC::SAM_GMC(const ParamManager &pm): SAM_Base(pm)
{
	thetaSamples.resize(D);
	for(int d=0; d<D; d++) thetaSamples[d].reserve(thBnin+thN); // for(auto &thetads: thetaSamples) thetads.resize(thBnin+thN, VectorXd::Zeros(K));
	btRej = 0;
	betaVd.resize(K, D);
	rejRates.reserve(Nmax);
	rejRates.emplace_back(D);
	for(int d=0; d<D; d++) rejRates.back()[d] = 1; // for(auto &el: rejRates.back()) el = 1;
	// write settings(.m)
	pm.Print(dirname+"settings", {"SAM_GMC"}, true);
	pm.Printm(dirname+"settings.m", {"SAM_GMC"}, true, dirname.substr(0, dirname.size()-1)+"_settings_GMC");
}
SAM_GMC::SAM_GMC(const string &dirname, const ParamManager &pm): SAM_Base(For::re, dirname, pm)
{
	thetaSamples.resize(D);
	for(int d=0; d<D; d++) thetaSamples[d].reserve(thBnin+thN);
	ifstream ifs(dirname+"betaRejRate");
	ifs >> btRej; ifs.close();
	betaVd.resize(K, D);
	rejRates.reserve(Nmax);
	ifs.open(dirname+"thetaRejRates");
	for(int i=0; i<betaSamples.size(); i++) {
		rejRates.emplace_back(D);
		for(int d=0; d<D; d++) ifs >> rejRates.back()[d]; // for(auto &el: rejRates.back()) ifs >> el;
	}
}

void SAM_GMC::InitTheta(void)
{
	const MatrixXd &beta = betaSamples.back();
	betaGram.noalias() = beta.transpose() * beta;
	MatrixXd betaGramInvBetaTrs = betaGram.ldlt().solve( beta.transpose() );
	for(int d=0; d<D; d++) {
		thetaSamples[d].clear();
		thetaSamples[d].emplace_back(K);
		VectorXd &thetad = thetaSamples[d].back();
		thetad.noalias() = betaGramInvBetaTrs * data[d];
		for(int k=0; k<K; k++) if(thetad(k) <= 0) thetad(k) = 1e-6;
		thetad /= thetad.lpNorm<1>();
	}
}

void SAM_GMC::Sample(const string &tsDirname)
{
	int pos; double localTime, checkpoint;
	SAM_Eval *ptr_tsModel = NULL;
	if(purpose == For::tr) {
		ptr_tsModel = new SAM_Eval((*this), pm);
		pos = 0; checkpoint = pm.d("ts_tmBeg");
	} else if (purpose == For::re) {
		ptr_tsModel = new SAM_Eval((*this), const_cast<ParamManager&>(pm), tsDirname);
		pos = betaSamples.size(); checkpoint = sampleTimes.back() + pm.d("ts_tmIntv");
	} else {
		cerr << "ERROR in SAM_GMC::Sample: wrong purpose!" << endl; throw;
	}
	SAM_Eval &tsModel = (*ptr_tsModel);
	const int bnin = pm.i("ts_bnin");
	GMC_spheres<SAM_GMC> gmcBeta((*this), betaSamples, &SAM_GMC::U_beta, &SAM_GMC::GradU_beta, pm.i("btL"), pm.d("btEps"));
	ProgressBar pgb(D, 40, 4, '|', 'v');

	// begin to sample!
	while(betaSamples.size() < Nmax) {
		cout << "Sample " << betaSamples.size() + 1 << "/" << Nmax << ":" << endl;
		localTime = omp_get_wtime();

		cout << "sampling theta for " << D << " documents..." << endl;
		this->InitTheta();
		rejRates.emplace_back(D);
		pgb.Reset();
#pragma omp parallel for
		for(int d=0; d<D; d++) {
			using namespace std::placeholders;
			GMC_simplex<SAM_GMC> gmcThetad( (*this), thetaSamples[d], bind(&SAM_GMC::U_thetad, _1, d), bind(&SAM_GMC::GradU_thetad, _1, _2, d), pm.i("thL"), pm.d("thEps") );
			betaVd.col(d).noalias() = betaSamples.back().transpose() * data[d];
			rejRates.back()[d] = gmcThetad.Sample(thBnin+thN) / (float)(thBnin+thN);
#pragma omp critical
			{
				pgb();
			}
		}

		cout << "sampling beta..." << flush;
		btRej *= betaSamples.size();
		btRej += gmcBeta.Sample(1);
		btRej /= betaSamples.size();
		cout << "done!" << endl;

		localTime = omp_get_wtime() - localTime;
		sampleTimes.emplace_back(sampleTimes.back() + localTime);
		cout << fixed << setprecision(3)
			 << "local time: " << localTime << ", total time: " << sampleTimes.back() << ", next checkpoint: " << checkpoint << endl;
		if(sampleTimes.back() > checkpoint) {
			this->WriteIncream(pos);
			if(betaSamples.size() < bnin) tsModel.bnin = 0;
			else if(betaSamples.size() < 2*bnin) tsModel.bnin = betaSamples.size() - bnin;
			else tsModel.bnin = bnin;
			tsModel.N = betaSamples.size() - tsModel.bnin;
			tsModel.EvalPerpSeq();
			checkpoint += pm.d("ts_tmIntv");
			pos = betaSamples.size();
		}
	}
	delete ptr_tsModel;
}

void SAM_GMC::WriteIncream(int pos)const
{
	this->SAM_Base::WriteIncream(pos);
	ofstream ofs(dirname+"betaRejRate");
	ofs << fixed << setprecision(3);
	ofs << btRej << endl; ofs.close();
	ofs.open(dirname+"thetaRejRates", ios::app);
	for(int i=pos; i<betaSamples.size(); i++) {
		for(float rij: rejRates[i]) ofs << " " << rij;
		ofs << endl;
	}
	ofs.close();
	cout << "Done!" << endl;
}

//////////////////////////

double SAM_GMC::U_beta(void)const
{
	VectorXd BThetaD(V);
	double U = 0;
	for(int n=thBnin; n<thBnin+thN; n++) {
		for(int d=0; d<D; d++) {
			BThetaD.noalias() = betaSamples.back() * thetaSamples[d][n];
			BThetaD.normalize();
			U -= data[d].dot( BThetaD );
		}
	}
	U *= kappa1 / (double)thN;

	this->MBarUpdate();
	U += Byme::LogVmfC( V, MBar.norm() );
	return U;
}
void SAM_GMC::GradU_beta(Ref<MatrixXd> grad)const
{
	grad.setZero();
#pragma omp parallel for
	for(int n=thBnin; n<thBnin+thN; n++) {
		double iBdNm;
		VectorXd BThetaD(V);
		MatrixXd localGrad(grad.rows(), grad.cols());
		localGrad.setZero();
		for(int d=0; d<D; d++) {
			BThetaD.noalias() = betaSamples.back() * thetaSamples[d][n];
			iBdNm = 1.0/BThetaD.norm();
			BThetaD *= - BThetaD.dot(data[d]) * (iBdNm * iBdNm * iBdNm);
			BThetaD += iBdNm * data[d];
			// now, BThetaD = iBdNm * v_d - b_d*(b_d^T * v_d)*(iBdNm^3)
			localGrad -= BThetaD * thetaSamples[d][n].transpose();
		}
#pragma omp critical
		{
			grad += localGrad;
		}
	}
	grad *= kappa1 / (double)thN;

	double iBdNm;
	this->MBarUpdate();
	iBdNm = MBar.norm();
	iBdNm = sigma * Byme::VmfA(V, iBdNm) / iBdNm;
	grad.colwise() += iBdNm * MBar;
}
double SAM_GMC::U_thetad(int d)const
{
	VectorXd BThetaD = betaSamples.back() * thetaSamples[d].back();
	BThetaD.normalize();
	return - kappa1 * BThetaD.dot(data[d]) - (alpha.array() - 1).matrix().dot( thetaSamples[d].back().array().log().matrix() );
}
void SAM_GMC::GradU_thetad(Ref<VectorXd> grad, int d)const
{
	const VectorXd &thetad = thetaSamples[d].back();
	double BdNm = (betaSamples.back() * thetad).norm();
	grad = -(alpha.array()-1).matrix().cwiseQuotient(thetad) - kappa1/BdNm * ( betaVd.col(d) - (thetad.dot(betaVd.col(d)) / BdNm / BdNm) * betaGram * thetad );
}

