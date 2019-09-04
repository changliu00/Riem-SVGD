#include <iostream>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include "SAM_SGGMC.hpp"
#include "../GMC.hpp"
#include "../SGGMC_inplace.hpp"
#include "../MyMath.hpp"
#include "../myinclude/Eigen/Cholesky" // ldlt

// SAM_SGGMC
SAM_SGGMC::SAM_SGGMC(const ParamManager &pm): SAM_Base(pm), S(pm.i("S")), max_iter(pm.i("max_iter"))
{
	thetaSamples.resize(Nmax);
	for(auto &thetam: thetaSamples) {
		thetam.resize(S);
		for(auto &thetams: thetam) thetams.reserve(thBnin+thN);
	}
	batchIds.resize(S);
	betaVs.resize(K, S);
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
	pm.Print(dirname+"settings", {"SAM_SGGMC"}, true);
	pm.Printm(dirname+"settings.m", {"SAM_SGGMC"}, true, dirname.substr(0, dirname.size()-1)+"_settings_SGGMC");
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
		rm.resize(S);
		for(int s=0; s<S; s++) rm[s] = 1; // for(auto &el: rejRates.back()) el = 1;
	}
}
SAM_SGGMC::SAM_SGGMC(const string &dirname, const ParamManager &pm): SAM_Base(For::re, dirname, pm), S(pm.i("S")), max_iter(pm.i("max_iter"))
{
	thetaSamples.resize(Nmax);
	for(auto &thetam: thetaSamples) {
		thetam.resize(S);
		for(auto &thetams: thetam) thetams.reserve(thBnin+thN);
	}
	batchIds.resize(S);
	betaVs.resize(K, S);
	ifstream ifs(dirname+"numEpochs");
	ifs >> epochs; ifs.close();
	// different treatment for "sampleTimes" and "rejRates"
	sampleTimes.clear(); sampleTimes.reserve(max_iter);
	rejRates.reserve(max_iter);
}

void SAM_SGGMC::InitTheta(int m)
{
	const MatrixXd &beta = betaSamples[m];
	betaGram.noalias() = beta.transpose() * beta;
	MatrixXd betaGramInvBetaTrs = betaGram.ldlt().solve( beta.transpose() );
	for(int s=0; s<S; s++) {
		thetaSamples[m][s].clear();
		thetaSamples[m][s].emplace_back(K);
		VectorXd &thetas = thetaSamples[m][s].back();
		thetas.noalias() = betaGramInvBetaTrs * data[batchIds[s]];
		for(int k=0; k<K; k++) if(thetas(k) <= 0) thetas(k) = 1e-6;
		thetas /= thetas.lpNorm<1>();
	}
}

void SAM_SGGMC::Sample_alongTime(const string &tsDirname)
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
		cerr << "ERROR in SAM_SGGMC::Sample: wrong purpose!" << endl; throw;
	}
	SAM_Eval &tsModel = (*ptr_tsModel);
	vector<SGGMC_inplace_spheres<SAM_SGGMC>> sggmcBetas;
	sggmcBetas.reserve(Nmax);
	using namespace std::placeholders;
	for(int m=0; m<Nmax; m++) sggmcBetas.emplace_back((*this), betaSamples[m], bind(&SAM_SGGMC::StoGradU_beta, _1, _2, m), D, pm.i("btL"), pm.d("btGm"), pm.d("btAl"));

	// begin to sample!
	while(epochs < max_iter) {
		++epochs;
		cout << "Epoch " << epochs << "/" << max_iter << ":" << endl;
		localTime = omp_get_wtime();

		cout << "Sampling theta..." << flush;
		Byme::RandPerm(batchIds.data(), D, S);
		rejRates.emplace_back(Nmax);
		auto &rej = rejRates.back();
		for(int m=0; m<Nmax; m++) {
			rej[m].resize(D);
			this->InitTheta(m);
#pragma omp parallel for
			for(int s=0; s<S; s++) {
				GMC_simplex<SAM_SGGMC> gmcThetas( (*this), thetaSamples[m][s], bind(&SAM_SGGMC::U_thetas, _1, m, s), bind(&SAM_SGGMC::GradU_thetas, _1, _2, m, s), pm.i("thL"), pm.d("thEps") );
				betaVs.col(s).noalias() = betaSamples[m].transpose() * data[batchIds[s]];
				rej[m][s] = gmcThetas.Sample(thBnin+thN) / (float)(thBnin+thN);
			}
		}
		cout << "done!" << endl;

		cout << "Sampling beta..." << flush;
#pragma omp parallel for
		for(int m=0; m<Nmax; m++) sggmcBetas[m].Sample(1);
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

void SAM_SGGMC::Sample_alongIter(const string &tsDirname)
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
		cerr << "ERROR in SAM_SGGMC::Sample: wrong purpose!" << endl; throw;
	}
	SAM_Eval &tsModel = (*ptr_tsModel);
	vector<SGGMC_inplace_spheres<SAM_SGGMC>> sggmcBetas;
	sggmcBetas.reserve(Nmax);
	using namespace std::placeholders;
	for(int m=0; m<Nmax; m++) sggmcBetas.emplace_back((*this), betaSamples[m], bind(&SAM_SGGMC::StoGradU_beta, _1, _2, m), D, pm.i("btL"), pm.d("btGm"), pm.d("btAl"));

	// begin to sample!
	while(epochs < max_iter) {
		++epochs;
		cout << "Epoch " << epochs << "/" << max_iter << ":" << endl;
		localTime = omp_get_wtime();

		cout << "Sampling theta..." << flush;
		Byme::RandPerm(batchIds.data(), D, S);
		rejRates.emplace_back(Nmax);
		auto &rej = rejRates.back();
		for(int m=0; m<Nmax; m++) {
			rej[m].resize(D);
			this->InitTheta(m);
#pragma omp parallel for
			for(int s=0; s<S; s++) {
				GMC_simplex<SAM_SGGMC> gmcThetas( (*this), thetaSamples[m][s], bind(&SAM_SGGMC::U_thetas, _1, m, s), bind(&SAM_SGGMC::GradU_thetas, _1, _2, m, s), pm.i("thL"), pm.d("thEps") );
				betaVs.col(s).noalias() = betaSamples[m].transpose() * data[batchIds[s]];
				rej[m][s] = gmcThetas.Sample(thBnin+thN) / (float)(thBnin+thN);
			}
		}
		cout << "done!" << endl;

		cout << "Sampling beta..." << flush;
#pragma omp parallel for
		for(int m=0; m<Nmax; m++) sggmcBetas[m].Sample(1);
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

void SAM_SGGMC::WriteIncream(void)const
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
	cout << "Done!" << endl;
}

//////////////////////////

void SAM_SGGMC::StoGradU_beta(Ref<MatrixXd> grad, int m)const
{
	VectorXd BThetaD(V);
	double iBdNm;
	grad.setZero();
	for(int n=thBnin; n<thBnin+thN; n++) {
		for(int s=0; s<S; s++) {
			BThetaD.noalias() = betaSamples[m] * thetaSamples[m][s][n];
			iBdNm = 1.0/BThetaD.norm();
			BThetaD *= - BThetaD.dot(data[batchIds[s]]) * (iBdNm * iBdNm * iBdNm);
			BThetaD += iBdNm * data[batchIds[s]];
			// now, BThetaD = iBdNm * v_d - b_d*(b_d^T * v_d)*(iBdNm^3)
			grad -= BThetaD * thetaSamples[m][s][n].transpose();
		}
	}
	grad *= kappa1 * D/(double)S/(double)thN;

	BThetaD = kappa0 * M + sigma * betaSamples[m].rowwise().sum();
	iBdNm = BThetaD.norm();
	iBdNm = sigma * Byme::VmfA(V, iBdNm) / iBdNm;
	grad.colwise() += iBdNm * BThetaD;
}
double SAM_SGGMC::U_thetas(int m, int s)const
{
	VectorXd BThetaD = betaSamples[m] * thetaSamples[m][s].back();
	BThetaD.normalize();
	return - kappa1 * BThetaD.dot(data[batchIds[s]]) - (alpha.array() - 1).matrix().dot( thetaSamples[m][s].back().array().log().matrix() );
}
void SAM_SGGMC::GradU_thetas(Ref<VectorXd> grad, int m, int s)const
{
	const VectorXd &thetas = thetaSamples[m][s].back();
	double BdNm = (betaSamples[m] * thetas).norm();
	grad = -(alpha.array()-1).matrix().cwiseQuotient(thetas) - kappa1/BdNm * ( betaVs.col(s) - (thetas.dot(betaVs.col(s)) / BdNm / BdNm) * betaGram * thetas );
}

