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
	thetaRejRates: betaSamples.size() * D
	betaRejRate: 1*Nmax
	numEpochs: 1*1
*****************/

/****
 * Nmax: num of particles
 * max_iter: num of maximum iterations
****/
class SAM_GMC : public SAM_Base {
public:
	const int max_iter;
	vector<vector<vector<VectorXd>>> thetaSamples; // Nmax * D * (thBnin+thN) * K
	vector<float> btRej; // Nmax
	MatrixXd betaVd;
	vector<vector<vector<float>>> rejRates;
public:
	SAM_GMC(const ParamManager &pm); // For::tr
	SAM_GMC(const string &dirname, const ParamManager &pm); // For::re
	void Sample_alongTime(const string &tsDirname = "");
	void Sample_alongIter(const string &tsDirname = "");
public: // private:
	void WriteIncream(void)const;
	void InitTheta(int m);
	double U_beta(int m)const;
	void GradU_beta(Ref<MatrixXd>, int m)const;
	// "d" is the position of the data point
	double U_thetad(int m, int d)const; // = -log p(\theta_d | \beta, v_d)
	void GradU_thetad(Ref<VectorXd>, int m, int d)const; // need to update "betaGram" and "betaVd" first
};

