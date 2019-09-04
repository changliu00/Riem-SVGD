#include <iostream>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include "SAM_GMC.hpp"
#include "../GMC.hpp"
#include "../GMC_inplace.hpp"
#include "../MyMath.hpp"
#include "../myinclude/Eigen/Cholesky" // ldlt

// SAM_GMC
SAM_GMC::SAM_GMC(const ParamManager &pm): SAM_Base(pm), max_iter(pm.i("max_iter"))
{
	thetaSamples.resize(Nmax);
	for(auto &thetam: thetaSamples) {
		thetam.resize(D);
		for(auto &thetamd: thetam) thetamd.reserve(thBnin+thN); // for(auto &thetads: thetaSamples) thetads.resize(thBnin+thN, VectorXd::Zeros(K));
	}
	btRej.resize(Nmax);
	for(float &el: btRej) el = 0;
	betaVd.resize(K, D);
	epochs = 1;
	// Initialize all the beta samples
	Bystd::RandNormal randN;
	betaSamples.resize(Nmax, MatrixXd(V,K));
	for(auto &beta: betaSamples) {
		randN(beta);
		beta /= sqrt(V);
		beta.colwise() += M;
		beta.colwise().normalize();
	}
	// write settings(.m)
	pm.Print(dirname+"settings", {"SAM_GMC"}, true);
	pm.Printm(dirname+"settings.m", {"SAM_GMC"}, true, dirname.substr(0, dirname.size()-1)+"_settings_GMC");
	// write "misc", which is global in this case
	ofstream ofs(dirname+"misc");
	ofs << Nmax << " " << D << " " << V << endl;
	ofs.close();
	// different treatment for "sampleTimes" and "rejRates"
	sampleTimes.reserve(max_iter);
	// sampleTimes.emplace_back(0); // This is done in SAM_Base::SAM_Base!
	rejRates.reserve(max_iter);
	rejRates.emplace_back(Nmax);
	for(auto &rm: rejRates.back()) {
		rm.resize(D);
		for(int d=0; d<D; d++) rm[d] = 1;
	}
}
SAM_GMC::SAM_GMC(const string &dirname, const ParamManager &pm): SAM_Base(For::re, dirname, pm), max_iter(pm.i("max_iter"))
{
	thetaSamples.resize(Nmax);
	for(auto &thetam: thetaSamples) {
		thetam.resize(D);
		for(auto &thetamd: thetam) thetamd.reserve(thBnin+thN); // for(auto &thetads: thetaSamples) thetads.resize(thBnin+thN, VectorXd::Zeros(K));
	}
	ifstream ifs(dirname+"betaRejRate");
	btRej.resize(Nmax);
	for(float &el: btRej) ifs >> el;
	ifs.close();
	betaVd.resize(K, D);
	ifs.open(dirname+"numEpochs");
	ifs >> epochs; ifs.close();
	// different treatment for "sampleTimes" and "rejRates"
	sampleTimes.clear(); sampleTimes.reserve(max_iter);
	rejRates.reserve(max_iter);
}

void SAM_GMC::InitTheta(int m)
{
	const MatrixXd &beta = betaSamples[m];
	betaGram.noalias() = beta.transpose() * beta;
	MatrixXd betaGramInvBetaTrs = betaGram.ldlt().solve( beta.transpose() );
	for(int d=0; d<D; d++) {
		thetaSamples[m][d].clear();
		thetaSamples[m][d].emplace_back(K);
		VectorXd &thetad = thetaSamples[m][d].back();
		thetad.noalias() = betaGramInvBetaTrs * data[d];
		for(int k=0; k<K; k++) if(thetad(k) <= 0) thetad(k) = 1e-6;
		thetad /= thetad.lpNorm<1>();
	}
}

void SAM_GMC::Sample_alongTime(const string &tsDirname)
{
	double lastTime, localTime, checkpoint;
	SAM_Eval *ptr_tsModel = NULL;
	if(purpose == For::tr) {
		ptr_tsModel = new SAM_Eval((*this), pm);
		checkpoint = pm.d("ts_tmBeg");
		lastTime = 0;
	} else if (purpose == For::re) {
		ptr_tsModel = new SAM_Eval((*this), const_cast<ParamManager&>(pm), tsDirname);
		ifstream ifs(dirname+"sampleTimes");
		string buffer;
		for(int i=0; i<epochs-1; i++) getline(ifs, buffer);
		ifs >> lastTime;
		checkpoint = lastTime + pm.d("ts_tmIntv");
	} else {
		cerr << "ERROR in SAM_GMC::Sample: wrong purpose!" << endl; throw;
	}
	SAM_Eval &tsModel = (*ptr_tsModel);
	vector<GMC_inplace_spheres<SAM_GMC>> gmcBetas;
	gmcBetas.reserve(Nmax);
	using namespace std::placeholders;
	for(int m=0; m<Nmax; m++) gmcBetas.emplace_back((*this), betaSamples[m], bind(&SAM_GMC::U_beta, _1, m), bind(&SAM_GMC::GradU_beta, _1, _2, m), pm.i("btL"), pm.d("btEps"));

	// begin to sample!
	while(epochs < max_iter) {
		++epochs;

		cout << "Epoch " << epochs << "/" << max_iter << ":" << endl;
		localTime = omp_get_wtime();

		cout << "Sampling theta..." << flush;
		rejRates.emplace_back(Nmax);
		auto &rej = rejRates.back();
		for(int m=0; m<Nmax; m++) {
			rej[m].resize(D);
			this->InitTheta(m);
#pragma omp parallel for
			for(int d=0; d<D; d++) {
				GMC_simplex<SAM_GMC> gmcThetad( (*this), thetaSamples[m][d], bind(&SAM_GMC::U_thetad, _1, m, d), bind(&SAM_GMC::GradU_thetad, _1, _2, m, d), pm.i("thL"), pm.d("thEps") );
				betaVd.col(d).noalias() = betaSamples[m].transpose() * data[d];
				rej[m][d] = gmcThetad.Sample(thBnin+thN) / (float)(thBnin+thN);
			}
		}
		cout << "done!" << endl;

		cout << "Sampling beta..." << flush;
#pragma omp parallel for
		for(int m=0; m<Nmax; m++) {
			btRej[m] *= epochs;
			btRej[m] += gmcBetas[m].Sample(1);
			btRej[m] /= epochs;
		}
		cout << "done!" << endl;

		localTime = omp_get_wtime() - localTime;
		lastTime += localTime;
		sampleTimes.emplace_back(lastTime);
		cout << fixed << setprecision(3)
			 << "local time: " << localTime << ", total time: " << lastTime << ", next checkpoint: " << checkpoint << endl;
		if(lastTime > checkpoint) {
			this->WriteIncream();
			tsModel.EvalPerpPara();
			checkpoint += pm.d("ts_tmIntv");
			sampleTimes.clear();
			rejRates.clear();
		}
	}
	delete ptr_tsModel;
}

void SAM_GMC::Sample_alongIter(const string &tsDirname)
{
	int checkpoint;
	double lastTime, localTime;
	SAM_Eval *ptr_tsModel = NULL;
	if(purpose == For::tr) {
		ptr_tsModel = new SAM_Eval((*this), pm);
		checkpoint = pm.i("ts_iterBeg");
		lastTime = 0;
	} else if (purpose == For::re) {
		ptr_tsModel = new SAM_Eval((*this), const_cast<ParamManager&>(pm), tsDirname);
		ifstream ifs(dirname+"sampleTimes");
		string buffer;
		for(int i=0; i<epochs-1; i++) getline(ifs, buffer);
		ifs >> lastTime;
		checkpoint = epochs + pm.i("ts_iterIntv");
	} else {
		cerr << "ERROR in SAM_GMC::Sample: wrong purpose!" << endl; throw;
	}
	SAM_Eval &tsModel = (*ptr_tsModel);
	vector<GMC_inplace_spheres<SAM_GMC>> gmcBetas;
	gmcBetas.reserve(Nmax);
	using namespace std::placeholders;
	for(int m=0; m<Nmax; m++) gmcBetas.emplace_back((*this), betaSamples[m], bind(&SAM_GMC::U_beta, _1, m), bind(&SAM_GMC::GradU_beta, _1, _2, m), pm.i("btL"), pm.d("btEps"));

	// begin to sample!
	while(epochs < max_iter) {
		++epochs;

		cout << "Epoch " << epochs << "/" << max_iter << ":" << endl;
		localTime = omp_get_wtime();

		cout << "Sampling theta..." << flush;
		rejRates.emplace_back(Nmax);
		auto &rej = rejRates.back();
		for(int m=0; m<Nmax; m++) {
			rej[m].resize(D);
			this->InitTheta(m);
#pragma omp parallel for
			for(int d=0; d<D; d++) {
				GMC_simplex<SAM_GMC> gmcThetad( (*this), thetaSamples[m][d], bind(&SAM_GMC::U_thetad, _1, m, d), bind(&SAM_GMC::GradU_thetad, _1, _2, m, d), pm.i("thL"), pm.d("thEps") );
				betaVd.col(d).noalias() = betaSamples[m].transpose() * data[d];
				rej[m][d] = gmcThetad.Sample(thBnin+thN) / (float)(thBnin+thN);
			}
		}
		cout << "done!" << endl;

		cout << "Updating beta..." << flush;
#pragma omp parallel for
		for(int m=0; m<Nmax; m++) {
			btRej[m] *= epochs;
			btRej[m] += gmcBetas[m].Sample(1);
			btRej[m] /= epochs;
		}
		cout << "done!" << endl;

		localTime = omp_get_wtime() - localTime;
		lastTime += localTime;
		sampleTimes.emplace_back(lastTime);
		cout << fixed << setprecision(3)
			 << "local time: " << localTime << ", total time: " << lastTime << ", next checkpoint: " << checkpoint << endl;
		if(epochs >= checkpoint) {
			this->WriteIncream();
			tsModel.EvalPerpPara();
			checkpoint += pm.i("ts_iterIntv");
			sampleTimes.clear();
			rejRates.clear();
		}
	}
	delete ptr_tsModel;
}

void SAM_GMC::WriteIncream(void)const
{
	ofstream ofs;
	cout << "DO NOT KILL NOW! Writing to disk..." << flush;
	ofs.open(dirname+"betaSamples"); // not "ios::app"! only last sample printed
	ofs << scientific << setprecision(6);
	for(const auto &beta: betaSamples) ofs << beta.transpose() << endl;
	ofs.close();
	ofs.open(dirname+"sampleTimes", ios::app);
	ofs << fixed << setprecision(3);
	for(double time: sampleTimes) ofs << time << endl;
	ofs.close();
	ofs.open(dirname+"numEpochs");
	ofs << epochs << endl; ofs.close();
	ofs.open(dirname+"thetaRejRates", ios::app);
	for(const auto &r: rejRates) {
		for(const auto &rm: r) {
			for(float rmd: rm) ofs << " " << rmd;
			ofs << endl;
		}
		ofs << endl;
	}
	ofs.close();
	ofs.open(dirname+"betaRejRate");
	ofs << fixed << setprecision(3);
	for(const auto &r: btRej) ofs << " " << r;
	ofs << endl; ofs.close();
	cout << "Done!" << endl;
}

//////////////////////////

double SAM_GMC::U_beta(int m)const
{
	VectorXd BThetaD(V);
	double U = 0;
	for(int n=thBnin; n<thBnin+thN; n++) {
		for(int d=0; d<D; d++) {
			BThetaD.noalias() = betaSamples[m] * thetaSamples[m][d][n];
			BThetaD.normalize();
			U -= data[d].dot( BThetaD );
		}
	}
	U *= kappa1 / (double)thN;

	BThetaD = kappa0 * M + sigma * betaSamples[m].rowwise().sum();
	U += Byme::LogVmfC( V, BThetaD.norm() );
	return U;
}
void SAM_GMC::GradU_beta(Ref<MatrixXd> grad, int m)const
{
	VectorXd BThetaD(V);
	double iBdNm;
	grad.setZero();
	for(int n=thBnin; n<thBnin+thN; n++) {
		for(int d=0; d<D; d++) {
			BThetaD.noalias() = betaSamples[m] * thetaSamples[m][d][n];
			iBdNm = 1.0/BThetaD.norm();
			BThetaD *= - BThetaD.dot(data[d]) * (iBdNm * iBdNm * iBdNm);
			BThetaD += iBdNm * data[d];
			// now, BThetaD = iBdNm * v_d - b_d*(b_d^T * v_d)*(iBdNm^3)
			grad -= BThetaD * thetaSamples[m][d][n].transpose();
		}
	}
	grad *= kappa1 / (double)thN;

	BThetaD = kappa0 * M + sigma * betaSamples[m].rowwise().sum();
	iBdNm = BThetaD.norm();
	iBdNm = sigma * Byme::VmfA(V, iBdNm) / iBdNm;
	grad.colwise() += iBdNm * BThetaD;
}
double SAM_GMC::U_thetad(int m, int d)const
{
	VectorXd BThetaD = betaSamples[m] * thetaSamples[m][d].back();
	BThetaD.normalize();
	return - kappa1 * BThetaD.dot(data[d]) - (alpha.array() - 1).matrix().dot( thetaSamples[m][d].back().array().log().matrix() );
}
void SAM_GMC::GradU_thetad(Ref<VectorXd> grad, int m, int d)const
{
	const VectorXd &thetad = thetaSamples[m][d].back();
	double BdNm = (betaSamples[m] * thetad).norm();
	grad = -(alpha.array()-1).matrix().cwiseQuotient(thetad) - kappa1/BdNm * ( betaVd.col(d) - (thetad.dot(betaVd.col(d)) / BdNm / BdNm) * betaGram * thetad );
}

