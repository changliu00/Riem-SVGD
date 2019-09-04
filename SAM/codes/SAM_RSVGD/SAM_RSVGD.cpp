#include <iostream>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include "SAM_RSVGD.hpp"
#include "../GMC.hpp"
#include "../RSVGD_SphProd.hpp"
#include "../MyMath.hpp"
#include "../myinclude/Eigen/Cholesky" // ldlt

// SAM_RSVGD
SAM_RSVGD::SAM_RSVGD(const ParamManager &pm): SAM_Base(pm), max_iter(pm.i("max_iter"))
{
	thetaSamples.resize(D);
	for(int d=0; d<D; d++) thetaSamples[d].reserve(thBnin+thN); // for(auto &thetads: thetaSamples) thetads.resize(thBnin+thN, VectorXd::Zeros(K));
	betaVd.resize(D, VectorXd(K));
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
	pm.Print(dirname+"settings", {"SAM_RSVGD"}, true);
	pm.Printm(dirname+"settings.m", {"SAM_RSVGD"}, true, dirname.substr(0, dirname.size()-1)+"_settings_RSVGD");
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
		for(int d=0; d<D; d++) rm[d] = 1; // for(auto &el: rejRates.back()) el = 1;
	}
}
SAM_RSVGD::SAM_RSVGD(const string &dirname, const ParamManager &pm): SAM_Base(For::re, dirname, pm), max_iter(pm.i("max_iter"))
{
	thetaSamples.resize(D);
	for(int d=0; d<D; d++) thetaSamples[d].reserve(thBnin+thN);
	betaVd.resize(D, VectorXd(K));
	ifstream ifs(dirname+"numEpochs");
	ifs >> epochs; ifs.close();
	// Initialize all the beta samples: done in SAM_Base::SAM_Base in For::re mode!
	// different treatment for "sampleTimes" and "rejRates"
	sampleTimes.clear(); sampleTimes.reserve(max_iter);
	rejRates.reserve(max_iter);
}

void SAM_RSVGD::InitTheta(int m)
{
	const MatrixXd &beta = betaSamples[m];
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

void SAM_RSVGD::Infer_alongTime(const string &tsDirname)
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
		cerr << "ERROR in SAM_RSVGD::Infer_alongTime: wrong purpose!" << endl; throw;
	}
	SAM_Eval &tsModel = (*ptr_tsModel);
	// tsModel.bnin = 0; tsModel.N = Nmax; // This is done in "SAM_Eval::EvalPerpPara".
	RSVGD_SpheresProd_Base<SAM_RSVGD> *rsvgdBeta;
	if(pm.s("btOptType") == "fixed")
		rsvgdBeta = new RSVGD_SpheresProd_fixed<SAM_RSVGD>((*this), betaSamples, &SAM_RSVGD::GradU_beta, pm.d("ibandw"), pm.d("btEps"));
	else if(pm.s("btOptType") == "rsgd")
		rsvgdBeta = new RSVGD_SpheresProd_rsgd<SAM_RSVGD>((*this), betaSamples, &SAM_RSVGD::GradU_beta, pm.d("ibandw"), pm.d("btEps"), pm.d("btKappa"), pm.i("btTau0"));
	else if(pm.s("btOptType") == "rsgdm")
		rsvgdBeta = new RSVGD_SpheresProd_rsgdm<SAM_RSVGD>((*this), betaSamples, &SAM_RSVGD::GradU_beta, pm.d("ibandw"), pm.d("btEps"), pm.d("btAlpha"));
	else if(pm.s("btOptType") == "rmsprop")
		rsvgdBeta = new RSVGD_SpheresProd_rmsprop<SAM_RSVGD>((*this), betaSamples, &SAM_RSVGD::GradU_beta, pm.d("ibandw"), pm.d("btEps"), pm.d("btAlpha"), pm.d("btLambda"));
	else {
		cerr << "ERROR in SAM_RSVGD::Infer_alongIter: unknown \"btOptType\" " << pm.s("btOptType") << "!" << endl; throw;
	}

	// begin to sample!
	while(epochs < max_iter) {
		++epochs;

		cout << "Epoch " << epochs << "/" << max_iter << ":" << endl;
		localTime = omp_get_wtime();
		cout << "updating beta samples..." << endl;
		rsvgdBeta->Update(1, dirname+"ibandws");
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
	delete ptr_tsModel, rsvgdBeta;
}

void SAM_RSVGD::Infer_alongIter(const string &tsDirname)
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
		cerr << "ERROR in SAM_RSVGD::Infer_alongIter: wrong purpose!" << endl; throw;
	}
	SAM_Eval &tsModel = (*ptr_tsModel);
	// tsModel.bnin = 0; tsModel.N = Nmax; // This is done in "SAM_Eval::EvalPerpPara".
	RSVGD_SpheresProd_Base<SAM_RSVGD> *rsvgdBeta;
	if(pm.s("btOptType") == "fixed")
		rsvgdBeta = new RSVGD_SpheresProd_fixed<SAM_RSVGD>((*this), betaSamples, &SAM_RSVGD::GradU_beta, pm.d("ibandw"), pm.d("btEps"));
	else if(pm.s("btOptType") == "rsgd")
		rsvgdBeta = new RSVGD_SpheresProd_rsgd<SAM_RSVGD>((*this), betaSamples, &SAM_RSVGD::GradU_beta, pm.d("ibandw"), pm.d("btEps"), pm.d("btKappa"), pm.i("btTau0"));
	else if(pm.s("btOptType") == "rsgdm")
		rsvgdBeta = new RSVGD_SpheresProd_rsgdm<SAM_RSVGD>((*this), betaSamples, &SAM_RSVGD::GradU_beta, pm.d("ibandw"), pm.d("btEps"), pm.d("btAlpha"));
	else if(pm.s("btOptType") == "rmsprop")
		rsvgdBeta = new RSVGD_SpheresProd_rmsprop<SAM_RSVGD>((*this), betaSamples, &SAM_RSVGD::GradU_beta, pm.d("ibandw"), pm.d("btEps"), pm.d("btAlpha"), pm.d("btLambda"));
	else {
		cerr << "ERROR in SAM_RSVGD::Infer_alongIter: unknown \"btOptType\" " << pm.s("btOptType") << "!" << endl; throw;
	}

	// begin to sample!
	while(epochs < max_iter) {
		++epochs;
		cout << "Epoch " << epochs << "/" << max_iter << ":" << endl;
		localTime = omp_get_wtime();
		cout << "updating beta samples..." << endl;
		rsvgdBeta->Update(1, dirname+"ibandws");
		cout << "done!" << endl;

		localTime = omp_get_wtime() - localTime;
		lastTime += localTime;
		sampleTimes.emplace_back(lastTime);
		cout << fixed << setprecision(3)
			 << "local time: " << localTime << ", total time: " << lastTime << endl;
		if(epochs >= checkpoint) {
			this->WriteIncream();
			tsModel.EvalPerpPara();
			checkpoint += pm.i("ts_iterIntv");
			sampleTimes.clear();
			rejRates.clear();
		}
	}
	delete ptr_tsModel, rsvgdBeta;
}

void SAM_RSVGD::WriteIncream(void)const
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

void SAM_RSVGD::GradU_beta(vector<MatrixXd> &grads)
{
	rejRates.emplace_back(Nmax);
	auto &rej = rejRates.back();
	for(int m=0; m<Nmax; m++) {
		this->InitTheta(m);
		rej[m].resize(D);
#pragma omp parallel for
		for(int d=0; d<D; d++) {
			using namespace std::placeholders;
			GMC_simplex<SAM_RSVGD> gmcThetad( (*this), thetaSamples[d], bind(&SAM_RSVGD::U_thetad, _1, m, d), bind(&SAM_RSVGD::GradU_thetad, _1, _2, m, d), pm.i("thL"), pm.d("thEps") );
			betaVd[d].noalias() = betaSamples[m].transpose() * data[d];
			rej[m][d] = gmcThetad.Sample(thBnin+thN) / (float)(thBnin+thN);
		}

		grads[m].setZero();
#pragma omp parallel for
		for(int n=thBnin; n<thBnin+thN; n++) {
			double iBdNm;
			VectorXd BThetaD(V);
			MatrixXd localGrad(V, K);
			localGrad.setZero();
			for(int d=0; d<D; d++) {
				BThetaD.noalias() = betaSamples[m] * thetaSamples[d][n];
				iBdNm = 1.0/BThetaD.norm();
				BThetaD *= - BThetaD.dot(data[d]) * (iBdNm * iBdNm * iBdNm);
				BThetaD += iBdNm * data[d];
				// now, BThetaD = iBdNm * v_d - b_d*(b_d^T * v_d)*(iBdNm^3)
				localGrad -= BThetaD * thetaSamples[d][n].transpose();
			}
#pragma omp critical
			{
				grads[m] += localGrad;
			}
		}
		grads[m] *= kappa1 / (double)thN;

		double iBdNm;
		MBar = kappa0 * M + sigma * betaSamples[m].rowwise().sum();
		iBdNm = MBar.norm();
		iBdNm = sigma * Byme::VmfA(V, iBdNm) / iBdNm;
		grads[m].colwise() += iBdNm * MBar;
	}
}
double SAM_RSVGD::U_thetad(int m, int d)const
{
	VectorXd BThetaD = betaSamples[m] * thetaSamples[d].back();
	BThetaD.normalize();
	return - kappa1 * BThetaD.dot(data[d]) - (alpha.array() - 1).matrix().dot( thetaSamples[d].back().array().log().matrix() );
}
void SAM_RSVGD::GradU_thetad(Ref<VectorXd> grad, int m, int d)const
{
	const VectorXd &thetad = thetaSamples[d].back();
	double BdNm = (betaSamples[m] * thetad).norm();
	grad = -(alpha.array()-1).matrix().cwiseQuotient(thetad) - kappa1/BdNm * ( betaVd[d] - (thetad.dot(betaVd[d]) / BdNm / BdNm) * betaGram * thetad );
}

