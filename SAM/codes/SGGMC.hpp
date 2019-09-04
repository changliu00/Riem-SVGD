#pragma once
#include <vector>
#include <functional> // std::function
#include <limits> // std::numeric_limits
#include "myinclude/Eigen/Dense"
#include "MyMath.hpp"
using namespace std;
using namespace Eigen;

template<class Model, class EigenVM = VectorXd>
class SGGMC
{
protected:
	Model &model;
	vector<EigenVM> &samples;
	int L; // number of steps
	double eps; // step size
	double sd; // sd of the Gaussian noise
	double damper; // exp(-C*eps*0.5) = exp(-alpha*0.5)

	typename vector<EigenVM>::iterator var;
	EigenVM vlc, grad, noise;
private:
	typedef function<void(const Model&, Ref<EigenVM>)> ER_C;
	ER_C StoGradPtnEnergy; // stochastic gradient of the potential energy
	virtual void Projection(void) = 0; // update "vlc" based on "*var"
	virtual void GeodFlowUpdate(void) = 0; // update "vlc" and "*var" based on "eps"
public:
	// D- total data size; gamma- the per-batch learning rate; alpha- for Coef (the coefficient of the friction); gradVariance- the variance of the stochastic gradient
	// "gradVariance < 2*D*alpha/gamma" is required!
	// "samples.back()" should not be changed after the sampler constructed! "this->Projection();" should be added to the constructor of every inheriting classes! 
	SGGMC(Model &model, vector<EigenVM> &samples, ER_C StoGradPtnEnergy, int D, int L = 1, double gamma = 0.01, double alpha = 0.01, double gradVariance = 0):
		model(model), samples(samples), StoGradPtnEnergy(StoGradPtnEnergy), L(L), 
		eps(sqrt(gamma/D)), sd(sqrt(2*alpha - gradVariance*eps*eps)), damper(exp(-alpha*0.5)),
		var(samples.end() - 1), vlc(var->rows(), var->cols()), grad(var->rows(), var->cols()), noise(var->rows(), var->cols()) {
			Bystd::RandNormal randN; randN(vlc);
		}
	void Reset(void) { var = samples.end() - 1; this->Projection(); }

	void Sample(int moreN) { // "samples.back()" should not be changed between two calls of this function!
		if(moreN <= 0) return;
		Bystd::RandNormal randN;
		var = samples.end() - 1;
		for(int i=0; i<moreN; i++) {
			samples.emplace_back(*var); // "samples" should be fully reserved to avoid reallocation! Otherwise the more consuming "samples.emplace_back(EigenVM(*var));" should be used!
			var = samples.end() - 1; // If the above condition holds, "var++;" also applies.

			eps *= 0.5; this->GeodFlowUpdate(); eps *= 2;
			StoGradPtnEnergy(model, grad);
			vlc = damper * (damper*vlc - eps*grad + sd*randN(noise));
			this->Projection();
			for(int j=1; j<L; j++) {
				this->GeodFlowUpdate();
				StoGradPtnEnergy(model, grad);
				vlc = damper * (damper*vlc - eps*grad + sd*randN(noise));
				this->Projection();
			}
			eps *= 0.5; this->GeodFlowUpdate(); eps *= 2;
		}
	}
};

// Derived classes
template<class Model>
class SGGMC_spheres : public SGGMC<Model, MatrixXd>
{
public:
	typedef function<void(const Model&, Ref<MatrixXd>)> ER_C;
	SGGMC_spheres(Model &model, vector<MatrixXd> &samples, ER_C StoGradPtnEnergy, int D, int L = 1, double gamma = 0.01, double alpha = 0.01, double gradVariance = 0): SGGMC<Model, MatrixXd>(model, samples, StoGradPtnEnergy, D, L, gamma, alpha, gradVariance) { this->Projection(); }
private:
	using SGGMC<Model, MatrixXd>::var;
	using SGGMC<Model, MatrixXd>::vlc;
	using SGGMC<Model, MatrixXd>::eps;

	void Projection(void) {
		for(int k=0; k<var->cols(); k++) vlc.col(k) -= vlc.col(k).dot(var->col(k)) * var->col(k);
	}
	void GeodFlowUpdate(void) {
		VectorXd BThetaD(var->rows());
		double vNorm, cosval, sinval;
		for(int k=0; k<var->cols(); k++) {
			vNorm = vlc.col(k).norm();
			cosval = cos(vNorm * eps);
			sinval = sin(vNorm * eps);
			BThetaD = vlc.col(k);
			vlc.col(k) = cosval * vlc.col(k) - vNorm*sinval * var->col(k);
			var->col(k) = cosval * var->col(k) + sinval/vNorm * BThetaD;
		}
	}
};

template<class Model>
class SGGMC_simplex : public SGGMC<Model, VectorXd>
{
public:
	typedef function<void(const Model&, Ref<VectorXd>)> ER_C;
	SGGMC_simplex(Model &model, vector<VectorXd> &samples, ER_C StoGradPtnEnergy, int D, int L = 1, double gamma = 0.01, double alpha = 0.01, double gradVariance = 0): SGGMC<Model, VectorXd>(model, samples, StoGradPtnEnergy, D, L, gamma, alpha, gradVariance) { this->Projection(); }
private:
	using SGGMC<Model, VectorXd>::var;
	using SGGMC<Model, VectorXd>::vlc;
	using SGGMC<Model, VectorXd>::eps;

	void Projection(void) {
		vlc.array() -= vlc.sum()/(double)vlc.size();
	}
	void GeodFlowUpdate(void) {
		double omg, kp, swp;
		int i, j, K = vlc.size();

		omg = eps;
		while(true) {
			kp = 2 * omg;
			j = 0;
			for(i=0; i<K; i++){
				if(vlc(i) < 0){
					swp = (*var)(i) / -vlc(i);
					if(swp < kp){
						kp = swp;
						j = i;
					}
				}
			}
			swp = omg<kp ? omg : kp;
			*var += swp * vlc;
			omg -= swp;
			if(omg > numeric_limits<double>::min()) {
				swp = (K * vlc(j) - vlc.sum()) * 2.0 / K / (K - 1);
				vlc.array() += swp;
				vlc(j) -= K * swp;
			} else break;
		}
	}
};

template<class Model>
class SGGMC_Stiefel : public SGGMC<Model, MatrixXd>
{
public:
	typedef function<void(const Model&, Ref<MatrixXd>)> ER_C;
	SGGMC_Stiefel(Model &model, vector<MatrixXd> &samples, ER_C StoGradPtnEnergy, int D, int L = 1, double gamma = 0.01, double alpha = 0.01, double gradVariance = 0): SGGMC<Model, MatrixXd>(model, samples, StoGradPtnEnergy, D, L, gamma, alpha, gradVariance) { this->Projection(); }
private:
	using SGGMC<Model, MatrixXd>::var;
	using SGGMC<Model, MatrixXd>::vlc;
	using SGGMC<Model, MatrixXd>::eps;

	void Projection(void) {
		vlc = vlc - 0.5*(*var)*( var->transpose() * vlc + vlc.transpose() * (*var) );
	}
	void GeodFlowUpdate(void) {
		int p = vlc.cols();
		MatrixXd A = (-eps) * var->transpose() * vlc, B(2*p, 2*p), 
				 expA(p, p), expB(2*p, 2*p), varRes = (*var);
		B << -A, (-eps) * vlc.transpose() * vlc,
			 eps*MatrixXd::Identity(p, p), -A;
		Byme::expm(expB, B);
		Byme::expm(expA, A);
		*var = varRes * expB.topLeftCorner(p,p) + vlc * expB.bottomLeftCorner(p,p);
		*var *= expA;
		vlc = varRes * expB.topRightCorner(p,p) + vlc * expB.bottomRightCorner(p,p);
		vlc *= expA;
		/*
		XV << (*var), vlc;
		XV = XV * expB * (MatrixXd(2*p, 2*p) << expA, MatrixXd::Zero(p, p), MatrixXd::Zero(p, p), expA).finished();
		*var = XV.leftCols(p); vlc = XV.rightCols(p);
		*/
	}
};

