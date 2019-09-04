#pragma once
// #define EIGEN_USE_MKL_ALL
#include <vector>
#include <string>
#include "myinclude/Eigen/Dense"
#include "myinclude/myutils_3.hpp"
using namespace std;
using namespace Eigen;
/*****************
File Formats:
Training outputs:
	settings(.m): output of ParamManager::Print(m)
	M: 1*V
	alpha: 1*K
	betaSamples: [K*V] * betaSamples.size()
	sampleTimes: betaSamples.size() * 1
	misc: betaSamples.size() D V
Test outputs:
	logperp_n/settings(.m): output of ParamManager::Print(m)
	logperp_n/results.m: [bnin, N, timepoint, log-perplexity]
Topwords:
	topwords_n: 
		K topn bnin N
		topic_No(from 0) word_1 ... word_topn
// Member variables of SAM_Base needed by
// 	topw: betaSamples
// 	test: betaSamples, sampleTimes, M, alpha
// 	retr: all
*****************/

class SAM_Base {
public:
	enum class For{tr, tw, ts, re};
	const For purpose;
	const string dirname; // '/' is trailed.
	const ParamManager &pm;
	const int V, D, K, Nmax, thBnin, thN; // D: num documents, K: num topics, Nmax: num max betaSamples
	double sigma, kappa0, kappa1; VectorXd M, alpha;
	vector<VectorXd> data;
	vector<MatrixXd> betaSamples;

	vector<double> sampleTimes;
	mutable VectorXd MBar; // V
	MatrixXd betaGram; // K*K
	int epochs; // added for parallel sample chains, instead of one sequantial sample chain.
public:
	SAM_Base(const ParamManager &pm); // for tr
	SAM_Base(For purpose, const string &dirname, const ParamManager &pm); // for other purposes. exception. "pm" is just some place to store parameters, no need to be initialized
	virtual void WriteIncream(int pos = 0)const; // betaSamples, misc
protected:
	void MBarUpdate()const { MBar = kappa0 * M + sigma * betaSamples.back().rowwise().sum(); }
};

class SAM_Eval {
public:
	enum class Type{invalid, llog, flog, lbaylog, fbaylog};
	const Type type;
	const SAM_Base &master;
	const string dirname; // '/' is trailed
	const int V, D, K, thN; // thN: the number of Dirichlet samples of theta for one document
	vector<VectorXd> data;
	int bnin, N;
public:
	SAM_Eval(const SAM_Base &master, const ParamManager &pm); // exception
	SAM_Eval(const SAM_Base &master, ParamManager &pm, const string &dirname); // For::re. exception
	double EvalPerpSeq(void);
	double EvalPerpPara(void); // added for parallel sample chains, instead of one sequantial sample chain.
protected:
	double perpRes;
	void EvalPerp_LiteLog(void);
	void EvalPerp_FullLog(void);
	void EvalPerp_LiteBayesianLog(void);
	void EvalPerp_FullBayesianLog(void);
};

class SAM_Topwords {
public:
	const SAM_Base &master;
	const int V, K;
	int topn, bnin, N;
private:
	MatrixXd beta;
	vector<string> dict;
public:
	SAM_Topwords(const SAM_Base &master, const ParamManager &pm); // load dict or exception
	void GetTopwords(void);
};

void LoadData(const string &filename, vector<VectorXd> &data, int &V, int &D); // throws exceptions

