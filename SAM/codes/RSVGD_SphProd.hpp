#pragma once
#include <vector>
#include <functional> // std::function
#include <limits> // std::numeric_limits
#include <algorithm> // std::nth_element
#include <omp.h>
#include "myinclude/Eigen/Core"
#include "MyMath.hpp"
using namespace std;
using namespace Eigen;

/*
template<class Model>
class RSVGD_spheres_fac
{
protected:
	Model &model;
	vector<MatrixXd> &samples;
	double eps; // step size
	double ibandw; // kernel(y,y') = exp(ibandw * y^T * y')

	int V, K, N; // #dim, #unit_vectors, #samples
	VectorXd vect;
	vector< MatrixXd > curr, grads;
	vector< vector<double> > ker, inprod, ygrad;
private:
	using VE_C = function<void(Model&, vector<MatrixXd>&)>;
	VE_C GradsLogp; // gradient of the PtnEnergy
public:
	RSVGD_spheres_fac(Model &model, vector<MatrixXd> &samples, VE_C GradsLogp, double eps, double ibandw):
		model(model), samples(samples), GradsLogp(GradsLogp), eps(eps), ibandw(ibandw),
		V(samples[0].rows()), K(samples[0].cols()), N(samples.size()), vect(V), grads(N, MatrixXd(V,K)),
		ker(N, vector<double>(N)), inprod(N, vector<double>(N)), ygrad(N, vector<double>(N))
	{
		double expIbandw = exp(ibandw);
		for(int i=0; i<N; i++) {
			inprod[i][i] = 1; ker[i][i] = expIbandw;
		}
	}

	void Update(int moreN) {
		for(int n=0; n<moreN; n++) {
			GradsLogp(model, grads);
			curr = samples;
			for(int k=0; k<K; k++) {
				for(int i=0; i<N; i++) {
					for(int j=0; j<i; j++) {
						inprod[i][j] = inprod[j][i]; ker[i][j] = ker[j][i];
					}
					for(int j=i+1; j<N; j++) {
						inprod[i][j] = curr[i].col(k).dot(curr[j].col(k));
						ker[i][j] = exp(ibandw * inprod[i][j]);
					}
					for(int j=0; j<N; j++) ygrad[i][j] = curr[i].col(k).dot(grads[j].col(k));
				}
				for(int i=0; i<N; i++) {
					vect.setZero();
					for(int j=0; j<N; j++) {
						double kerij = ker[i][j], inprodij = inprod[i][j];
						vect += ibandw * kerij * grads[j].col(k) + ibandw * ibandw * kerij * (ibandw + ygrad[i][j] - (ygrad[j][j] + V - 1) * (inprodij + 1.0/ibandw) - 2*inprodij - ibandw*inprodij*inprodij) * curr[j].col(k);
					}
					vect /= N;
					// projection
					vect = vect - vect.dot(curr[i].col(k)) * curr[i].col(k);
					// geodesic update
					double vNorm = vect.norm();
					samples[i].col(k) = cos(vNorm*eps) * curr[i].col(k) + sin(vNorm*eps)/vNorm * vect;
				}
			}
		}
	}
};
*/

template<class Model>
class RSVGD_SpheresProd_Base
{
protected:
	Model &model;
	vector<MatrixXd> &samples;
	const bool isAdapt;

	using RowArrayXd = Array<double, 1, Dynamic>;
	int V, K, N; // #dim, #unit_vectors, #samples
	RowArrayXd ibandws; // kernel(y,y') = exp(ibandw * y^T * y')
	vector< MatrixXd > grads;
	vector< ArrayXXd > vects;
	vector< vector<double> > ker;
	vector< vector<RowArrayXd> > inprod_vec;
	vector< RowArrayXd > ygraddiag;
	vector< vector<double> > containers;

	using VE_C = function<void(Model&, vector<MatrixXd>&)>;
	VE_C GradsLogp; // gradient of the PtnEnergy
public:
	RSVGD_SpheresProd_Base(Model &model, vector<MatrixXd> &samples, VE_C GradsLogp, double ibandw):
		model(model), samples(samples), GradsLogp(GradsLogp), isAdapt(ibandw < 0),
		V(samples[0].rows()), K(samples[0].cols()), N(samples.size()),
		ibandws(K), grads(N, MatrixXd(V,K)), vects(N, ArrayXXd(V,K)),
		ker(N, vector<double>(N)), inprod_vec(N, vector<RowArrayXd>(N)), ygraddiag(N)
	{
		for(int i=0; i<N; i++) {
			inprod_vec[i][i] = RowArrayXd::Ones(K);
		}
		if(!isAdapt) {
			double expIbandw = exp(ibandw * K);
			for(int i=0; i<N; i++) ker[i][i] = expIbandw;
			ibandws.setConstant(ibandw);
		} else containers.resize(K, vector<double>(N*(N-1)/2));
	}

	void GetVects(const string &filename) {
		bool writeLog = (filename != "");
		ofstream ofs;
		GradsLogp(model, grads);
		for(int i=0; i<N; i++) {
			for(int j=0; j<i; j++) {
				inprod_vec[i][j] = inprod_vec[j][i];
			}
			ygraddiag[i] = samples[i].cwiseProduct(grads[i]).colwise().sum();
			for(int j=i+1; j<N; j++) {
				inprod_vec[i][j] = samples[i].cwiseProduct(samples[j]).colwise().sum();
			}
		}
		if(isAdapt) {
#pragma omp parallel for
			for(int k=0; k<K; k++) {
				int n = 0;
				vector<double> &contk = containers[k];
				for(int i=0; i<N; i++)
					for(int j=0; j<i; j++)
						contk[n++] = inprod_vec[i][j](k);
				nth_element(contk.begin(), contk.begin() + contk.size()/2, contk.end());
				ibandws(k) = log(N-1) / (1 - contk[contk.size()/2]);
			}
			if(writeLog) {
				ofs.open(filename, ios::app);
				ofs << ibandws << endl;
				ofs.close();
			}
			double expIbandw = exp(ibandws.sum());
			for(int i=0; i<N; i++) ker[i][i] = expIbandw;
		}
		for(int i=0; i<N; i++) {
			for(int j=0; j<i; j++) {
				ker[i][j] = ker[j][i];
			}
			for(int j=i+1; j<N; j++) {
				ker[i][j] = exp(ibandws.matrix().dot(inprod_vec[i][j].matrix()));
			}
		}
#pragma omp parallel for
		for(int i=0; i<N; i++) {
			vects[i].setZero();
			for(int j=0; j<N; j++) {
				RowArrayXd ibandwInprod = ibandws * inprod_vec[i][j];
				double mult = (j==i)? (ygraddiag[j].matrix().dot(ibandws.matrix())) : (samples[i].cwiseProduct(grads[j]).colwise().sum().matrix().dot(ibandws.matrix()));
				mult += ibandws.matrix().squaredNorm() - ibandwInprod.matrix().squaredNorm() - (ygraddiag[j]+(V-1)).matrix().dot(ibandwInprod.matrix());
				vects[i] += samples[j].array().rowwise() * (((mult-V+1) - (2*ibandwInprod + ygraddiag[j])) * ibandws * ker[i][j]) + grads[j].array().rowwise() * (ker[i][j] * ibandws);
			}
			vects[i] /= N;
			// projection
			vects[i] -= samples[i].array().rowwise() * ((samples[i].array() * vects[i]).colwise().sum());
		}
	}

	virtual void Update(int moreN, const string &filename = "") = 0;
};

template<class Model>
class RSVGD_SpheresProd_fixed : public RSVGD_SpheresProd_Base<Model>
{
protected:
	const double eps;
	Array<double, 1, Dynamic> norms;
	using VE_C = function<void(Model&, vector<MatrixXd>&)>;
	using RSVGD_SpheresProd_Base<Model>::K;
	using RSVGD_SpheresProd_Base<Model>::N;
	using RSVGD_SpheresProd_Base<Model>::samples;
	using RSVGD_SpheresProd_Base<Model>::vects;
public:
	RSVGD_SpheresProd_fixed(Model &model, vector<MatrixXd> &samples, VE_C GradsLogp, double ibandw, double eps):
		RSVGD_SpheresProd_Base<Model>(model, samples, GradsLogp, ibandw),
		eps(eps), norms(K) {};

	void Update(int moreN, const string &filename) {
		for(int n=0; n<moreN; n++) {
			this->GetVects(filename);
			for(int i=0; i<N; i++) {
				// geodesic update
				norms = vects[i].matrix().colwise().norm();
				samples[i] = samples[i].array().rowwise() * ((eps*norms).cos()) + vects[i].rowwise() * ((eps*norms).sin() / norms);
			}
		}
	}
};

template<class Model>
class RSVGD_SpheresProd_rsgd : public RSVGD_SpheresProd_Base<Model>
{
protected:
	int iter;
	const double eps0, kappa; // 0.5 < kappa <= 1
	double eps;
	Array<double, 1, Dynamic> norms;
	using VE_C = function<void(Model&, vector<MatrixXd>&)>;
	using RSVGD_SpheresProd_Base<Model>::K;
	using RSVGD_SpheresProd_Base<Model>::N;
	using RSVGD_SpheresProd_Base<Model>::samples;
	using RSVGD_SpheresProd_Base<Model>::vects;
public:
	RSVGD_SpheresProd_rsgd(Model &model, vector<MatrixXd> &samples, VE_C GradsLogp, double ibandw, double eps0 = 1, double kappa = 1, int tau0 = 0):
		RSVGD_SpheresProd_Base<Model>(model, samples, GradsLogp, ibandw),
		iter(tau0), eps0(eps0), kappa(kappa), norms(K) {};

	void Update(int moreN, const string &filename) {
		for(int n=0; n<moreN; n++) {
			++iter;
			eps = eps0 / pow(iter, kappa);
			this->GetVects(filename);
			for(int i=0; i<N; i++) {
				// geodesic update
				norms = vects[i].matrix().colwise().norm();
				samples[i] = samples[i].array().rowwise() * ((eps*norms).cos()) + vects[i].rowwise() * ((eps*norms).sin() / norms);
			}
		}
	}
};

template<class Model>
class RSVGD_SpheresProd_rsgdm : public RSVGD_SpheresProd_Base<Model>
{
protected:
	const double eps, alpha;
	Array<double, 1, Dynamic> norms, cosAngles, sinAngles;
	vector< ArrayXXd > momentum;
	ArrayXXd sample_i;
	using VE_C = function<void(Model&, vector<MatrixXd>&)>;
	using RSVGD_SpheresProd_Base<Model>::V;
	using RSVGD_SpheresProd_Base<Model>::K;
	using RSVGD_SpheresProd_Base<Model>::N;
	using RSVGD_SpheresProd_Base<Model>::samples;
	using RSVGD_SpheresProd_Base<Model>::vects;
public:
	RSVGD_SpheresProd_rsgdm(Model &model, vector<MatrixXd> &samples, VE_C GradsLogp, double ibandw, double eps, double alpha = 0.9):
		RSVGD_SpheresProd_Base<Model>(model, samples, GradsLogp, ibandw),
		eps(eps), alpha(alpha), norms(K), cosAngles(K), sinAngles(K), momentum(N, ArrayXXd(V,K)), sample_i(V,K)
	{
		for(int i=0; i<N; i++) momentum[i].setZero();
	}

	void Update(int moreN, const string &filename) {
		for(int n=0; n<moreN; n++) {
			this->GetVects(filename);
			for(int i=0; i<N; i++) {
				momentum[i] = alpha * momentum[i] + eps * vects[i];
				// geodesic update
				norms = momentum[i].matrix().colwise().norm();
				cosAngles = (eps*norms).cos();
				sinAngles = (eps*norms).sin();
				sample_i = samples[i];
				samples[i] = samples[i].array().rowwise() * cosAngles + momentum[i].rowwise() * (sinAngles / norms);
				momentum[i] = momentum[i].rowwise() * cosAngles - sample_i.rowwise() * (sinAngles * norms);
			}
		}
	}
};

template<class Model>
class RSVGD_SpheresProd_rmsprop : public RSVGD_SpheresProd_Base<Model>
{
protected:
	const double eps, alpha, lambda;
	Array<double, 1, Dynamic> norms, cosAngles, sinAngles;
	vector< ArrayXXd > accum_vects;
	ArrayXXd sample_i, para_perp;
	using VE_C = function<void(Model&, vector<MatrixXd>&)>;
	using RSVGD_SpheresProd_Base<Model>::V;
	using RSVGD_SpheresProd_Base<Model>::K;
	using RSVGD_SpheresProd_Base<Model>::N;
	using RSVGD_SpheresProd_Base<Model>::samples;
	using RSVGD_SpheresProd_Base<Model>::vects;
public:
	RSVGD_SpheresProd_rmsprop(Model &model, vector<MatrixXd> &samples, VE_C GradsLogp, double ibandw, double eps, double alpha = 0.9, double lambda = 1e-5):
		RSVGD_SpheresProd_Base<Model>(model, samples, GradsLogp, ibandw),
		eps(eps), alpha(alpha), lambda(lambda), norms(K), cosAngles(K), sinAngles(K), accum_vects(N, ArrayXXd(V,K)), sample_i(V,K), para_perp(V,K)
	{
		for(int i=0; i<N; i++) accum_vects[i].setZero();
	}

	void Update(int moreN, const string &filename) {
		for(int n=0; n<moreN; n++) {
			this->GetVects(filename);
			for(int i=0; i<N; i++) {
				accum_vects[i] = alpha * accum_vects[i] + (1-alpha) * vects[i].pow(2);
				vects[i] /= lambda + accum_vects[i].sqrt();
				vects[i] -= samples[i].array().rowwise() * ((samples[i].array() * vects[i]).colwise().sum());
				// geodesic update
				norms = vects[i].matrix().colwise().norm();
				cosAngles = (eps*norms).cos();
				sinAngles = (eps*norms).sin();
				sample_i = samples[i];
				samples[i] = samples[i].array().rowwise() * cosAngles + vects[i].rowwise() * (sinAngles / norms);
				para_perp = accum_vects[i] - sample_i.rowwise() * ((sample_i * accum_vects[i]).colwise().sum());
				para_perp -= vects[i].rowwise() * ((vects[i]*para_perp).colwise().sum());
				accum_vects[i] = (accum_vects[i]-para_perp).rowwise() * cosAngles - sample_i.rowwise() * (sinAngles * norms) + para_perp;
			}
		}
	}
};

/*
template<class Model>
class RSVGD_SpheresProd_rsvrg : public RSVGD_SpheresProd_Base<Model>
{
protected:
	const double eps, alpha, lambda;
	Array<double, 1, Dynamic> norms, cosAngles, sinAngles;
	vector< ArrayXXd > accum_vects;
	ArrayXXd sample_i, para_perp;
	using VE_C = function<void(Model&, vector<MatrixXd>&)>;
	using RSVGD_SpheresProd_Base<Model>::V;
	using RSVGD_SpheresProd_Base<Model>::K;
	using RSVGD_SpheresProd_Base<Model>::N;
	using RSVGD_SpheresProd_Base<Model>::samples;
	using RSVGD_SpheresProd_Base<Model>::vects;
public:
	RSVGD_SpheresProd_rmsprop(Model &model, vector<MatrixXd> &samples, VE_C GradsLogp, double ibandw, double eps, double alpha = 0.9, double lambda = 1e-5):
		RSVGD_SpheresProd_Base<Model>(model, samples, GradsLogp, ibandw),
		eps(eps), alpha(alpha), lambda(lambda), norms(K), cosAngles(K), sinAngles(K), accum_vects(N, ArrayXXd(V,K)), sample_i(V,K), para_perp(V,K)
	{
		for(int i=0; i<N; i++) accum_vects[i].setZero();
	}

	void Update(int moreN) {
		for(int n=0; n<moreN; n++) {
			this->GetVects();
			for(int i=0; i<N; i++) {
				accum_vects[i] = alpha * accum_vects[i] + (1-alpha) * vects[i].pow(2);
				vects[i] /= lambda + accum_vects[i].sqrt();
				vects[i] -= samples[i].array().rowwise() * ((samples[i].array() * vects[i]).colwise().sum());
				// geodesic update
				norms = vects[i].matrix().colwise().norm();
				cosAngles = (eps*norms).cos();
				sinAngles = (eps*norms).sin();
				sample_i = samples[i];
				samples[i] = samples[i].array().rowwise() * cosAngles + vects[i].rowwise() * (sinAngles / norms);
				para_perp = accum_vects[i] - sample_i.rowwise() * ((sample_i * accum_vects[i]).colwise().sum());
				para_perp -= vects[i].rowwise() * ((vects[i]*para_perp).colwise().sum());
				accum_vects[i] = (accum_vects[i]-para_perp).rowwise() * cosAngles - sample_i.rowwise() * (sinAngles * norms) + para_perp;
			}
		}
	}
};
*/

/*
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
*/

/*
template<class Model>
class RSVGD_simplex : public RSVGD<Model, VectorXd>
{
public:
	typedef function<double(const Model&)> D__C;
	typedef function<void(const Model&, Ref<VectorXd>)> VE_C;
	RSVGD_simplex(Model &model, vector<VectorXd> &samples, D__C PtnEnergy, VE_C GradsLogp, int L, double eps): RSVGD<Model, VectorXd>(model, samples, PtnEnergy, GradsLogp, L, eps) {};
private:
	using RSVGD<Model, VectorXd>::var;
	using RSVGD<Model, VectorXd>::vlc;
	using RSVGD<Model, VectorXd>::eps;

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
*/

