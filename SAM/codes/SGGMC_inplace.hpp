#pragma once
#include <vector>
#include <functional> // std::function
#include <limits> // std::numeric_limits
#include "myinclude/Eigen/Dense"
#include "MyMath.hpp"
using namespace std;
using namespace Eigen;

template<class Model, class EigenVM = VectorXd>
class SGGMC_inplace
{
protected:
	Model &model;
	EigenVM &sample;
	int L; // number of steps
	double eps; // step size
	double sd; // sd of the Gaussian noise
	double damper; // exp(-C*eps*0.5) = exp(-alpha*0.5)

	EigenVM vlc, grad, noise;
private:
	typedef function<void(const Model&, Ref<EigenVM>)> ER_C;
	ER_C StoGradPtnEnergy; // stochastic gradient of the potential energy
	virtual void Projection(void) = 0; // update "vlc" based on "sample"
	virtual void GeodFlowUpdate(void) = 0; // update "vlc" and "sample" based on "eps"
public:
	// D- total data size; gamma- the per-batch learning rate; alpha- for Coef (the coefficient of the friction); gradVariance- the variance of the stochastic gradient
	// "gradVariance < 2*D*alpha/gamma" is required!
	// "sample" should be properly initialized before constructing this sampler. "this->Projection();" should be added to the constructor of every inheriting classes! 
	SGGMC_inplace(Model &model, EigenVM &sample, ER_C StoGradPtnEnergy, int D, int L = 1, double gamma = 0.01, double alpha = 0.01, double gradVariance = 0):
		model(model), sample(sample), StoGradPtnEnergy(StoGradPtnEnergy), L(L), 
		eps(sqrt(gamma/D)), sd(sqrt(2*alpha - gradVariance*eps*eps)), damper(exp(-alpha*0.5)),
		vlc(sample.rows(), sample.cols()), grad(sample.rows(), sample.cols()), noise(sample.rows(), sample.cols()) {
			Bystd::RandNormal randN; randN(vlc);
		}

	void Sample(int moreN) { // "sample" should not be changed between two calls of this function!
		if(moreN <= 0) return;
		Bystd::RandNormal randN;
		for(int i=0; i<moreN; i++) {
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
class SGGMC_inplace_spheres : public SGGMC_inplace<Model, MatrixXd>
{
public:
	typedef function<void(const Model&, Ref<MatrixXd>)> ER_C;
	SGGMC_inplace_spheres(Model &model, MatrixXd &sample, ER_C StoGradPtnEnergy, int D, int L = 1, double gamma = 0.01, double alpha = 0.01, double gradVariance = 0): SGGMC_inplace<Model, MatrixXd>(model, sample, StoGradPtnEnergy, D, L, gamma, alpha, gradVariance) { this->Projection(); }
private:
	using SGGMC_inplace<Model, MatrixXd>::sample;
	using SGGMC_inplace<Model, MatrixXd>::vlc;
	using SGGMC_inplace<Model, MatrixXd>::eps;

	void Projection(void) {
		for(int k=0; k<sample.cols(); k++) vlc.col(k) -= vlc.col(k).dot(sample.col(k)) * sample.col(k);
	}
	void GeodFlowUpdate(void) {
		VectorXd BThetaD(sample.rows());
		double vNorm, cosval, sinval;
		for(int k=0; k<sample.cols(); k++) {
			vNorm = vlc.col(k).norm();
			cosval = cos(vNorm * eps);
			sinval = sin(vNorm * eps);
			BThetaD = vlc.col(k);
			vlc.col(k) = cosval * vlc.col(k) - vNorm*sinval * sample.col(k);
			sample.col(k) = cosval * sample.col(k) + sinval/vNorm * BThetaD;
		}
	}
};

template<class Model>
class SGGMC_inplace_simplex : public SGGMC_inplace<Model, VectorXd>
{
public:
	typedef function<void(const Model&, Ref<VectorXd>)> ER_C;
	SGGMC_inplace_simplex(Model &model, VectorXd &sample, ER_C StoGradPtnEnergy, int D, int L = 1, double gamma = 0.01, double alpha = 0.01, double gradVariance = 0): SGGMC_inplace<Model, VectorXd>(model, sample, StoGradPtnEnergy, D, L, gamma, alpha, gradVariance) { this->Projection(); }
private:
	using SGGMC_inplace<Model, VectorXd>::sample;
	using SGGMC_inplace<Model, VectorXd>::vlc;
	using SGGMC_inplace<Model, VectorXd>::eps;

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
					swp = sample(i) / -vlc(i);
					if(swp < kp){
						kp = swp;
						j = i;
					}
				}
			}
			swp = omg<kp ? omg : kp;
			sample += swp * vlc;
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
class SGGMC_inplace_Stiefel : public SGGMC_inplace<Model, MatrixXd>
{
public:
	typedef function<void(const Model&, Ref<MatrixXd>)> ER_C;
	SGGMC_inplace_Stiefel(Model &model, MatrixXd &sample, ER_C StoGradPtnEnergy, int D, int L = 1, double gamma = 0.01, double alpha = 0.01, double gradVariance = 0): SGGMC_inplace<Model, MatrixXd>(model, sample, StoGradPtnEnergy, D, L, gamma, alpha, gradVariance) { this->Projection(); }
private:
	using SGGMC_inplace<Model, MatrixXd>::sample;
	using SGGMC_inplace<Model, MatrixXd>::vlc;
	using SGGMC_inplace<Model, MatrixXd>::eps;

	void Projection(void) {
		vlc = vlc - 0.5*sample*( sample.transpose() * vlc + vlc.transpose() * sample );
	}
	void GeodFlowUpdate(void) {
		int p = vlc.cols();
		MatrixXd A = (-eps) * sample.transpose() * vlc, B(2*p, 2*p), 
				 expA(p, p), expB(2*p, 2*p), varRes = sample;
		B << -A, (-eps) * vlc.transpose() * vlc,
			 eps*MatrixXd::Identity(p, p), -A;
		Byme::expm(expB, B);
		Byme::expm(expA, A);
		sample = varRes * expB.topLeftCorner(p,p) + vlc * expB.bottomLeftCorner(p,p);
		sample *= expA;
		vlc = varRes * expB.topRightCorner(p,p) + vlc * expB.bottomRightCorner(p,p);
		vlc *= expA;
		/*
		XV << sample, vlc;
		XV = XV * expB * (MatrixXd(2*p, 2*p) << expA, MatrixXd::Zero(p, p), MatrixXd::Zero(p, p), expA).finished();
		sample = XV.leftCols(p); vlc = XV.rightCols(p);
		*/
	}
};

