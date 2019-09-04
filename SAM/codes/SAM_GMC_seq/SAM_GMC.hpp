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
	thetaRejRates: betaSamples.size() * D
	betaRejRate: btRej
*****************/

class SAM_GMC : public SAM_Base {
public:
	vector<vector<VectorXd>> thetaSamples; // D * (thBnin+thN) * K
	float btRej;
	MatrixXd betaVd;
	vector<vector<float>> rejRates;
public:
	SAM_GMC(const ParamManager &pm); // For::tr
	SAM_GMC(const string &dirname, const ParamManager &pm); // For::re
	void Sample(const string &tsDirname = "");
public: // private:
	void WriteIncream(int pos = 0)const;
	void InitTheta(void);
	double U_beta(void)const;
	void GradU_beta(Ref<MatrixXd>)const;
	// "d" is the position of the data point
	double U_thetad(int d)const; // = -log p(\theta_d | \beta, v_d)
	void GradU_thetad(Ref<VectorXd>, int d)const; // need to update "betaGram" and "betaVd" first
};

