#include <iostream>
#include <iomanip>
#include <omp.h>
#include "SAM_SGGMC.hpp"
#include "../GMC.hpp"
#include "../SGGMC.hpp"
#include "../MyMath.hpp"

// SAM_SGGMC
SAM_SGGMC::SAM_SGGMC(const ParamManager &pm): SAM_Base(pm), S(pm.i("S"))
{
	thetaSamples.resize(S);
	for(int s=0; s<S; s++) thetaSamples[s].reserve(thBnin+thN);
	batchIds.resize(S);
	betaVs.resize(K, S);
	rejRates.reserve(Nmax);
	rejRates.emplace_back(S);
	for(int s=0; s<S; s++) rejRates.back()[s] = 1;
	// write settings(.m)
	pm.Print(dirname+"settings", {"SAM_SGGMC"}, true);
	pm.Printm(dirname+"settings.m", {"SAM_SGGMC"}, true, dirname.substr(0, dirname.size()-1)+"_settings_SGGMC");
}
SAM_SGGMC::SAM_SGGMC(const string &dirname, const ParamManager &pm): SAM_Base(For::re, dirname, pm), S(pm.i("S"))
{
	thetaSamples.resize(S);
	for(int s=0; s<S; s++) thetaSamples[s].reserve(thBnin+thN);
	batchIds.resize(S);
	betaVs.resize(K, S);
	rejRates.reserve(Nmax);
	ifstream ifs(dirname+"thetaRejRates");
	for(int i=0; i<betaSamples.size(); i++) {
		rejRates.emplace_back(S);
		for(int s=0; s<S; s++) ifs >> rejRates.back()[s];
	}
}

void SAM_SGGMC::InitTheta(void)
{
	const MatrixXd &beta = betaSamples.back();
	betaGram.noalias() = beta.transpose() * beta;
	MatrixXd betaGramInvBetaTrs = betaGram.ldlt().solve( beta.transpose() );
	for(int s=0; s<S; s++) {
		thetaSamples[s].clear();
		thetaSamples[s].emplace_back(K);
		VectorXd &thetas = thetaSamples[s].back();
		thetas.noalias() = betaGramInvBetaTrs * data[batchIds[s]];
		for(int k=0; k<K; k++) if(thetas(k) <= 0) thetas(k) = 1e-6;
		thetas /= thetas.lpNorm<1>();
	}
}

void SAM_SGGMC::Sample(const string &tsDirname)
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
		cerr << "ERROR in SAM_SGGMC::Sample: wrong purpose!" << endl; throw;
	}
	SAM_Eval &tsModel = (*ptr_tsModel);
	const int N = pm.i("ts_N");
	SGGMC_spheres<SAM_SGGMC> sggmcBeta((*this), betaSamples, &SAM_SGGMC::StoGradU_beta, D, pm.i("btL"), pm.d("btGm"), pm.d("btAl"));
	ProgressBar pgb(S, 40, 4, '|', 'v');

	// begin to sample!
	while(betaSamples.size() < Nmax) {
		cout << "Sample " << betaSamples.size() + 1 << "/" << Nmax << ":" << endl;
		localTime = omp_get_wtime();

		cout << "sampling theta for " << S << " documents..." << endl;
		Byme::RandPerm(batchIds.data(), D, S);
		this->InitTheta();
		rejRates.emplace_back(S);
		pgb.Reset();
#pragma omp parallel for
		for(int s=0; s<S; s++) {
			using namespace std::placeholders;
			GMC_simplex<SAM_SGGMC> gmcThetas( (*this), thetaSamples[s], bind(&SAM_SGGMC::U_thetas, _1, s), bind(&SAM_SGGMC::GradU_thetas, _1, _2, s), pm.i("thL"), pm.d("thEps") );
			betaVs.col(s).noalias() = betaSamples.back().transpose() * data[batchIds[s]];
			rejRates.back()[s] = gmcThetas.Sample(thBnin+thN) / (float)(thBnin+thN);
#pragma omp critical
			{
				pgb();
			}
		}

		cout << "sampling beta..." << flush;
		sggmcBeta.Sample(1);
		cout << "done!" << endl;

		localTime = omp_get_wtime() - localTime;
		sampleTimes.emplace_back(sampleTimes.back() + localTime);
		cout << fixed << setprecision(3)
			 << "local time: " << localTime << ", total time: " << sampleTimes.back() << ", next checkpoint: " << checkpoint << endl;
		if(sampleTimes.back() > checkpoint) {
			this->WriteIncream(pos);
			if(betaSamples.size() < N) tsModel.bnin = 0;
			else tsModel.bnin = betaSamples.size() - N;
			tsModel.N = betaSamples.size() - tsModel.bnin;
			tsModel.EvalPerpSeq();
			checkpoint += pm.d("ts_tmIntv");
			pos = betaSamples.size();
		}
	}
	delete ptr_tsModel;
}

void SAM_SGGMC::WriteIncream(int pos)const
{
	this->SAM_Base::WriteIncream(pos);
	ofstream ofs(dirname+"thetaRejRates", ios::app);
	ofs << fixed << setprecision(3);
	for(int i=pos; i<betaSamples.size(); i++) {
		for(float rij: rejRates[i]) ofs << " " << rij;
		ofs << endl;
	}
	ofs.close();
	cout << "Done!" << endl;
}

//////////////////////////

void SAM_SGGMC::StoGradU_beta(Ref<MatrixXd> grad)const
{
	grad.setZero();
#pragma omp parallel for
	for(int n=thBnin; n<thBnin+thN; n++) {
		double iBdNm;
		VectorXd BThetaD(V);
		MatrixXd localGrad(grad.rows(), grad.cols());
		localGrad.setZero();
		for(int s=0; s<S; s++) {
			BThetaD.noalias() = betaSamples.back() * thetaSamples[s][n];
			iBdNm = 1.0/BThetaD.norm();
			BThetaD *= - BThetaD.dot(data[batchIds[s]]) * (iBdNm * iBdNm * iBdNm);
			BThetaD += iBdNm * data[batchIds[s]];
			// now, BThetaD = iBdNm * v_d - b_d*(b_d^T * v_d)*(iBdNm^3)
			localGrad -= BThetaD * thetaSamples[s][n].transpose();
		}
#pragma omp critical
		{
			grad += localGrad;
		}
	}
	grad *= kappa1 * D/(double)S/(double)thN;

	double iBdNm;
	this->MBarUpdate();
	iBdNm = MBar.norm();
	iBdNm = sigma * Byme::VmfA(V, iBdNm) / iBdNm;
	grad.colwise() += iBdNm * MBar;
}
double SAM_SGGMC::U_thetas(int s)const
{
	VectorXd BThetaD = betaSamples.back() * thetaSamples[s].back();
	BThetaD.normalize();
	return - kappa1 * BThetaD.dot(data[batchIds[s]]) - (alpha.array() - 1).matrix().dot( thetaSamples[s].back().array().log().matrix() );
}
void SAM_SGGMC::GradU_thetas(Ref<VectorXd> grad, int s)const
{
	const VectorXd &thetas = thetaSamples[s].back();
	double BdNm = (betaSamples.back() * thetas).norm();
	grad = -(alpha.array()-1).matrix().cwiseQuotient(thetas) - kappa1/BdNm * ( betaVs.col(s) - (thetas.dot(betaVs.col(s)) / BdNm / BdNm) * betaGram * thetas );
}

