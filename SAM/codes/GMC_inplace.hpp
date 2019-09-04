#pragma once
#include <vector>
#include <functional> // std::function
#include <limits> // std::numeric_limits
#include "myinclude/Eigen/Core"
#include "MyMath.hpp"
using namespace std;
using namespace Eigen;

template<class Model, class EigenVM = VectorXd>
class GMC_inplace
{
protected:
	Model &model;
	EigenVM &sample;
	int L; // number of steps
	double eps; // step size
	EigenVM old, vlc, grad;
private:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<EigenVM>)> ER_C;
	D__C PtnEnergy; // potential energy
	ER_C GradPtnEnergy; // gradient of the PtnEnergy
	virtual void Projection(void) = 0; // update "vlc" based on current "sample"
	virtual void GeodFlowUpdate(void) = 0; // update "vlc" and "sample" based on "eps"
public:
	GMC_inplace(Model &model, EigenVM &sample, D__C PtnEnergy, ER_C GradPtnEnergy, int L, double eps):
		model(model), sample(sample), PtnEnergy(PtnEnergy), GradPtnEnergy(GradPtnEnergy), L(L), eps(eps), 
		old(sample.rows(), sample.cols()), vlc(sample.rows(), sample.cols()), grad(sample.rows(), sample.cols()) {};

	int Sample(int moreN) { // The model should be appropriately initialized before calling this function. Return the number of rejections.
		if(moreN <= 0) return 0;
		int numRej = 0;
		double h0, rejRate;
		Bystd::RandNormal randN;
		for(int i=0; i<moreN; i++) {
			old = sample;
			randN(vlc); this->Projection();
			h0 = PtnEnergy(model) + 0.5 * vlc.squaredNorm();
				GradPtnEnergy(model, grad);
				vlc -= 0.5*eps * grad;
				this->Projection();
				this->GeodFlowUpdate();
			for(int j=1; j<L; j++) {
				GradPtnEnergy(model, grad);
				vlc -= eps * grad;
				this->Projection();
				this->GeodFlowUpdate();
			}
				GradPtnEnergy(model, grad);
				vlc -= 0.5*eps * grad;
				this->Projection();
			rejRate = exp(h0 - PtnEnergy(model) - 0.5 * vlc.squaredNorm());
			if(rejRate < 1 && Byme::RandUnif01() > rejRate) { // reject
				sample = old; numRej++;
			}
		}
		return numRej;
	}
};

// Derived classes
template<class Model>
class GMC_inplace_spheres : public GMC_inplace<Model, MatrixXd>
{
public:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<MatrixXd>)> ER_C;
	GMC_inplace_spheres(Model &model, MatrixXd &sample, D__C PtnEnergy, ER_C GradPtnEnergy, int L, double eps): GMC_inplace<Model, MatrixXd>(model, sample, PtnEnergy, GradPtnEnergy, L, eps) {};
private:
	using GMC_inplace<Model, MatrixXd>::sample;
	using GMC_inplace<Model, MatrixXd>::vlc;
	using GMC_inplace<Model, MatrixXd>::eps;

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
class GMC_inplace_simplex : public GMC_inplace<Model, VectorXd>
{
public:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<VectorXd>)> ER_C;
	GMC_inplace_simplex(Model &model, vector<VectorXd> &sample, D__C PtnEnergy, ER_C GradPtnEnergy, int L, double eps): GMC_inplace<Model, VectorXd>(model, sample, PtnEnergy, GradPtnEnergy, L, eps) {};
private:
	using GMC_inplace<Model, VectorXd>::sample;
	using GMC_inplace<Model, VectorXd>::vlc;
	using GMC_inplace<Model, VectorXd>::eps;

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
class GMC_inplace_Stiefel : public GMC_inplace<Model, MatrixXd>
{
public:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<MatrixXd>)> ER_C;
	GMC_inplace_Stiefel(Model &model, MatrixXd &sample, D__C PtnEnergy, ER_C GradPtnEnergy, int L, double eps): GMC_inplace<Model, MatrixXd>(model, sample, PtnEnergy, GradPtnEnergy, L, eps) {};
private:
	using GMC_inplace<Model, MatrixXd>::sample;
	using GMC_inplace<Model, MatrixXd>::vlc;
	using GMC_inplace<Model, MatrixXd>::eps;

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

