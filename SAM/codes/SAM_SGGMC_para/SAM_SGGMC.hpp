#pragma once
// #define EIGEN_USE_MKL_ALL
#include <vector>
#include <string>
#include "../SAM_Base.hpp"
#include "../myinclude/Eigen/Core"
#include "../myinclude/myutils_3.hpp"
using namespace std;
using namespace Eigen;
/*****************
File Formats:
Training outputs:
	// thetaLast: 1*K
	thetaRejRates: betaSamples.size() * S
	numEpochs: 1*1
*****************/

/****
 * Nmax: num of particles
 * max_iter: num of maximum iterations
****/
class SAM_SGGMC : public SAM_Base {
public:
	const int S;
	const int max_iter;
	vector<vector<vector<VectorXd>>> thetaSamples; // Nmax * S * (thBnin+thN) * K
	vector<int> batchIds; // S
	MatrixXd betaVs; // K*S
	vector<vector<vector<float>>> rejRates; // buffSize(<max_iter) * Nmax * S. Only stores rejRates from last write to current, not globally stored. "sampleTimes" is also treated like this.
public:
	SAM_SGGMC(const ParamManager &pm); // For::tr
	SAM_SGGMC(const string &dirname, const ParamManager &pm); // For::re
	void Sample_alongTime(const string &tsDirname = "");
	void Sample_alongIter(const string &tsDirname = "");
public: // private:
	void WriteIncream(void)const;
	void InitTheta(int m);
	void StoGradU_beta(Ref<MatrixXd>, int m)const;
	// "s" is the position of a subset data point
	double U_thetas(int m, int s)const; // = -log p(\theta_d | \beta, v_d)
	void GradU_thetas(Ref<VectorXd>, int m, int s)const; // need to update "betaGram" and "betaVs" first
};

