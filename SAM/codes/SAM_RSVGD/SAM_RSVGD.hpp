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
	thetaRejRates: epochs * D
	numEpochs: 1*1
*****************/

/****
 * Nmax: num of particles
 * max_iter: num of maximum iterations
****/
class SAM_RSVGD : public SAM_Base {
public:
	const int max_iter;
	vector<vector<VectorXd>> thetaSamples; // D * (thBnin+thN) * K
	vector<vector<vector<float>>> rejRates; // buffSize(<max_iter) * Nmax * D. Only stores rejRates from last write to current, not globally stored. "sampleTimes" is also treated like this.
	vector<VectorXd> betaVd;
public:
	SAM_RSVGD(const ParamManager &pm); // For::tr
	SAM_RSVGD(const string &dirname, const ParamManager &pm); // For::re
	void Infer_alongTime(const string &tsDirname = "");
	void Infer_alongIter(const string &tsDirname = "");
public: // private:
	void WriteIncream(void)const;
	void InitTheta(int m);
	void GradU_beta(vector<MatrixXd>&);
	// "d" is the position of the data point
	double U_thetad(int m, int d)const; // = -log p(\theta_d | \beta, v_d)
	void GradU_thetad(Ref<VectorXd>, int m, int d)const; // need to update "betaGram" and "betaVd" first
};

