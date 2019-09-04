#pragma once
// #define EIGEN_USE_MKL_ALL
#include <vector>
#include <string>
#include "../SAM_Base.hpp"
#include "../myinclude/Eigen/Dense"
#include "../myinclude/myutils_3.hpp"
using namespace std;
using namespace Eigen;
/*****************
File Formats:
Training outputs:
	// thetaLast: 1*K
	thetaRejRates: betaSamples.size() * S
*****************/

class SAM_SGGMC : public SAM_Base {
public:
	const int S;
	vector<vector<VectorXd>> thetaSamples; // S * (thBnin+thN) * K
	vector<int> batchIds; // S
	MatrixXd betaVs; // K*S
	vector<vector<float>> rejRates;
public:
	SAM_SGGMC(const ParamManager &pm); // For::tr
	SAM_SGGMC(const string &dirname, const ParamManager &pm); // For::re
	void Sample(const string &tsDirname = "");
public: // private:
	void WriteIncream(int pos = 0)const;
	void InitTheta(void);
	void StoGradU_beta(Ref<MatrixXd>)const;
	// "s" is the position of a subset data point
	double U_thetas(int s)const; // = -log p(\theta_d | \beta, v_d)
	void GradU_thetas(Ref<VectorXd>, int s)const; // need to update "betaGram" and "betaVs" first
};

