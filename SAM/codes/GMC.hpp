#pragma once
#include <vector>
#include <functional> // std::function
#include <limits> // std::numeric_limits
#include "myinclude/Eigen/Dense"
#include "MyMath.hpp"
using namespace std;
using namespace Eigen;

template<class Model, class EigenVM = VectorXd>
class GMC
{
protected:
	Model &model;
	vector<EigenVM> &samples;
	int L; // number of steps
	double eps; // step size

	typename vector<EigenVM>::iterator var;
	EigenVM vlc, grad;
private:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<EigenVM>)> ER_C;
	D__C PtnEnergy; // potential energy
	ER_C GradPtnEnergy; // gradient of the PtnEnergy
	virtual void Projection(void) = 0; // update "vlc" based on "*var"
	virtual void GeodFlowUpdate(void) = 0; // update "vlc" and "*var" based on "eps"
public:
	GMC(Model &model, vector<EigenVM> &samples, D__C PtnEnergy, ER_C GradPtnEnergy, int L, double eps):
		model(model), samples(samples), PtnEnergy(PtnEnergy), GradPtnEnergy(GradPtnEnergy), L(L), eps(eps), 
		var(samples.end() - 1), vlc(var->rows(), var->cols()), grad(var->rows(), var->cols()) {};

	int Sample(int moreN) { // The model should be appropriately initialized before calling this function. Return the number of rejections.
		if(moreN <= 0) return 0;
		int numRej = 0;
		double h0, rejRate;
		Bystd::RandNormal randN;
		var = samples.end() - 1;
		for(int i=0; i<moreN; i++) {
			samples.emplace_back(*var); // "samples" should be fully reserved to avoid reallocation! Otherwise the more consuming "samples.emplace_back(EigenVM(*var));" should be used!
			var = samples.end() - 1; // If the above condition holds, "var++;" also applies.
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
				*var = *(var - 1); numRej++;
			}
		}
		return numRej;
	}
};

// Derived classes
template<class Model>
class GMC_spheres : public GMC<Model, MatrixXd>
{
public:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<MatrixXd>)> ER_C;
	GMC_spheres(Model &model, vector<MatrixXd> &samples, D__C PtnEnergy, ER_C GradPtnEnergy, int L, double eps): GMC<Model, MatrixXd>(model, samples, PtnEnergy, GradPtnEnergy, L, eps) {};
private:
	using GMC<Model, MatrixXd>::var;
	using GMC<Model, MatrixXd>::vlc;
	using GMC<Model, MatrixXd>::eps;

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
class GMC_simplex : public GMC<Model, VectorXd>
{
public:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<VectorXd>)> ER_C;
	GMC_simplex(Model &model, vector<VectorXd> &samples, D__C PtnEnergy, ER_C GradPtnEnergy, int L, double eps): GMC<Model, VectorXd>(model, samples, PtnEnergy, GradPtnEnergy, L, eps) {};
private:
	using GMC<Model, VectorXd>::var;
	using GMC<Model, VectorXd>::vlc;
	using GMC<Model, VectorXd>::eps;

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
class GMC_Stiefel : public GMC<Model, MatrixXd>
{
public:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<MatrixXd>)> ER_C;
	GMC_Stiefel(Model &model, vector<MatrixXd> &samples, D__C PtnEnergy, ER_C GradPtnEnergy, int L, double eps): GMC<Model, MatrixXd>(model, samples, PtnEnergy, GradPtnEnergy, L, eps) {};
private:
	using GMC<Model, MatrixXd>::var;
	using GMC<Model, MatrixXd>::vlc;
	using GMC<Model, MatrixXd>::eps;

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

