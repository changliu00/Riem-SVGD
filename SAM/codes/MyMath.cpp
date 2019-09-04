#include "MyMath.hpp"
#include <stdio.h> // printf


///////////////////////
//                   //
//   Miscellaneous   //
//                   //
///////////////////////

// for the VMF distribution
#include <boost/math/special_functions/bessel.hpp> // for cyl_bessel_i
// #include <boost/math/special_functions/bessel_prime.hpp> // for cyl_bessel_i_prime, but the compiler on the server cannot find it. So I'll write the derivative function myself.

// the inverse of the normalization constant of VMF distribution
double Byboost::VmfC(int D, double k)
{
	double halfD, res;
	
	halfD = (double)D / 2.0;
	try{
		res = (double)( ( pow(k/(2*Byme::pi), halfD) )/( (long double)k * boost::math::cyl_bessel_i((long double)(halfD-1), (long double)k) ) );
	}catch(int er){
		printf("VmfC: exception %d met with arguments (%d, %.4le)\n", er, D, k);
		throw er;
	}catch(...){
		printf("VmfC: exception met with arguments (%d, %.4le)\n", D, k);
		throw (int)0;
	}
	if(!std::isnormal(res)){
		printf("VmfC: exception 1 met with arguments (%d, %.4le): res = %lf\n", D, k, res);
		throw (int)1;
	}
	return res;
}
// the logarithm of inverse of the normalization constant of VMF distribution
double Byboost::LogVmfC(int D, double k)
{
	double halfD, res;

	halfD = (double)D / 2.0;
	try{
		res = (halfD-1) * log(k) - halfD * log(2*Byme::pi) - (double)log( boost::math::cyl_bessel_i((long double)(halfD-1), (long double)k) );
	}catch(int er){
		printf("LogVmfC: exception %d met with arguments (%d, %.4le)\n", er, D, k);
		throw er;
	}catch(...){
		printf("LogVmfC: exception met with arguments (%d, %.4le)\n", D, k);
		throw (int)0;
	}
	if(!std::isfinite(res)){
		printf("LogVmfC: exception 1 met with arguments (%d, %.4le): res = %lf\n", D, k, res);
		throw (int)1;
	}
	return res;
}
// the derivative of bessel_i
double Byboost::my_cyl_bessel_i_prime(double D, double k)
{
	double res;

	try{
		if(D != 0)
			res = (double)(( boost::math::cyl_bessel_i((long double)(D-1), (long double)k) + boost::math::cyl_bessel_i((long double)(D+1), (long double)k) ) / 2.0);
		else
			res = (double)( boost::math::cyl_bessel_i((long double)1, (long double)k) );
	}catch(int er){
		printf("my_cyl_bessel_i_prime: exception %d met with arguments (%.4le, %.4le)\n", er, D, k);
		throw er;
	}catch(...){
		printf("my_cyl_bessel_i_prime: exception met with arguments (%.4le, %.4le)\n", D, k);
		throw (int)0;
	}
	if(!std::isnormal(res)){
		printf("my_cyl_bessel_i_prime: exception 1 met with arguments (%.4le, %.4le): res = %lf\n", D, k, res);
		throw (int)1;
	}
	return res;
}
// the derivative of log(VmfC())
double Byboost::VmfA(int D, double k)
{
	double halfD, res;

	halfD = (double)D * 0.5;
	try{
		res = - (double)( boost::math::cyl_bessel_i((long double)halfD, (long double)k) / 
			boost::math::cyl_bessel_i((long double)(halfD-1), (long double)k) );
	}catch(int er){
		printf("VmfA: exception %d met with arguments (%d, %.4le)\n", er, D, k);
		throw er;
	}catch(...){
		printf("VmfA: exception met with arguments (%d, %.4le)\n", D, k);
		throw (int)0;
	}
	if(!std::isnormal(res)){
		printf("VmfA: exception 1 met with arguments (%d, %.4le): res = %lf\n", D, k, res);
		throw (int)1;
	}
	return res;
}



// basic LogBesselI's
double Byme::LogBesselISmallXEps(double a, double x, double eps)
{
	int m = 0;
	double res, resOld, aLog, twoLog, mLogFac, maLogFac;
	aLog = log(x/2.0);
	twoLog = 2*aLog;
	aLog *= a;
	mLogFac = 0;
	maLogFac = lgamma(a+1);

	res = aLog - mLogFac - maLogFac;
	do{
		m++;
		resOld = res;
		aLog += twoLog;
		mLogFac += log(m);
		maLogFac += log(m+a);
		res = LogSum( res, aLog - mLogFac - maLogFac );
	}while( fabs((res-resOld)/res) > eps );

	return res;
}
double Byme::LogBesselISmallXEpsTest(double a, double x, double eps, bool verbose)
{
	int m = 0;
	double res, resOld, aLog, twoLog, mLogFac, maLogFac;
	aLog = log(x/2.0);
	twoLog = 2*aLog;
	aLog *= a;
	mLogFac = 0;
	maLogFac = lgamma(a+1);

	res = aLog - mLogFac - maLogFac;
	if(verbose)
		printf("%d: %.13le\n", m, res);
	do{
		m++;
		resOld = res;
		aLog += twoLog;
		mLogFac += log(m);
		maLogFac += log(m+a);
		res = LogSum( res, aLog - mLogFac - maLogFac );
		if(verbose)
			printf("%d: %.13le, %.9le\n", m, res, fabs((res-resOld)/res));
	}while( fabs((res-resOld)/res) > eps );
	printf("final m: %d\n", m);

	return res;
}
double Byme::LogBesselISmallXN(double a, double x, int n)
{
	double res, aLog, twoLog, mLogFac, maLogFac;
	aLog = log(x/2.0);
	twoLog = 2*aLog;
	aLog *= a;
	mLogFac = 0;
	maLogFac = lgamma(a+1);

	res = aLog - mLogFac - maLogFac;
	for(int m=1; m<=n; m++){
		aLog += twoLog;
		mLogFac += log(m);
		maLogFac += log(m+a);
		res = LogSum( res, aLog - mLogFac - maLogFac );
	}
	return res;
}
double Byme::LogBesselISmallXNTest(double a, double x, int n)
{
	double res, aLog, twoLog, mLogFac, maLogFac;
	aLog = log(x/2.0);
	twoLog = 2*aLog;
	aLog *= a;
	mLogFac = 0;
	maLogFac = lgamma(a+1);

	res = aLog - mLogFac - maLogFac;
	printf("0: %.13le\n", res);
	for(int m=1; m<=n; m++){
		aLog += twoLog;
		mLogFac += log(m);
		maLogFac += log(m+a);
		res = LogSum( res, aLog - mLogFac - maLogFac );
		printf("%d: %.13le\n", m, res);
	}
	return res;
}


// basic LogBesselIBigX's
double Byme::LogBesselIBigXEps(double a, double x, double eps)
{
	int m = 0, maxM = floor(a);
	double Am = 1, ASum = 1, fourASq = 4*a*a,
		   mult = 1.0/(8*x);
	do{
		m++;
		Am *= ( -fourASq + (2*m-1)*(2*m-1) ) * mult / m;
		ASum += Am;
	}while( (ASum < 0 || fabs(Am)/ASum > eps) && (m < maxM) );
	return log(ASum) + x - log(2 * Byme::pi * x) * 0.5;
}
double Byme::LogBesselIBigXEpsTest(double a, double x, double eps, bool verbose)
{
	int m = 0, maxM = floor(a);
	double Am = 1, ASum = 1, fourASq = 4*a*a,
		   mult = 1.0/(8*x),
		   res = x - log(2 * Byme::pi * x) * 0.5;
	if(verbose)
		printf("0: %.13le\n", res);
	do{
		m++;
		Am *= ( -fourASq + (2*m-1)*(2*m-1) ) * mult / m;
		ASum += Am;
		if(verbose)
			printf( "%d: %.13le\n", m, res + log(ASum) );
	}while( (ASum < 0 || fabs(Am)/ASum > eps) && (m < maxM) );
	printf("final m: %d\n", m);

	return res + log(ASum);
}
double Byme::LogBesselIBigXN(double a, double x, int n)
{
	double Am = 1, ASum = 1, fourASq = 4*a*a,
		   mult = 1.0/(8*x);
	for(int m=1; m<=n; m++){
		Am *= ( -fourASq + (2*m-1)*(2*m-1) ) * mult / m;
		ASum += Am;
	}
	return log(ASum) + x - log(2 * Byme::pi * x) * 0.5;
}
double Byme::LogBesselIBigXNTest(double a, double x, int n)
{
	double Am = 1, ASum = 1, fourASq = 4*a*a,
		   mult = 1.0/(8*x),
		   res = x - log(2 * Byme::pi * x) * 0.5;
	printf("0: %.13le\n", res);
	for(int m=1; m<=n; m++){
		Am *= ( -fourASq + (2*m-1)*(2*m-1) ) * mult / m;
		ASum += Am;
		printf( "%d: %.13le\n", m, res + log(ASum) );
	}
	return res + log(ASum);
}


/*******************
double Byme::LogBesselI(double a, double x)
{
	int m = 0;
	double res, resOld, aLog, twoLog, mLogFac, maLogFac;
	aLog = log(x/2.0);
	twoLog = 2*aLog;
	aLog *= a;
	mLogFac = 0;
	maLogFac = lgamma(a+1);

	res = aLog - mLogFac - maLogFac;
	do{
		m++;
		resOld = res;
		aLog += twoLog;
		mLogFac += log(m);
		maLogFac += log(m+a);
		res = LogSum( res, aLog - mLogFac - maLogFac );
	}while( fabs((res-resOld)/res) > BymeLogBesselI_DefaultEps );

	return res;
}
double Byme::LogBesselI(double a, double x)
{
	double res;

	try{
		res = boost::math::cyl_bessel_i(a, x);
	}catch(...){
		res = INFINITY; // INFINITY declared in cmath
	}
	if(res == -INFINITY){
		int m = 0;
		double resOld, aLog, twoLog, mLogFac, maLogFac;
		aLog = log(x/2.0);
		twoLog = 2*aLog;
		aLog *= a;
		mLogFac = 0;
		maLogFac = lgamma(a+1);

		res = aLog - mLogFac - maLogFac;
		do{
			m++;
			resOld = res;
			aLog += twoLog;
			mLogFac += log(m);
			maLogFac += log(m+a);
			res = LogSum( res, aLog - mLogFac - maLogFac );
		}while( fabs((res-resOld)/res) > BymeLogBesselI_DefaultEps && m <= BymeLogBesselI_DefaultMMax );
	}
	return res;
}

double Byme::LogBesselIPrime(double a, double x)
{
	if(a != 0){
		return LogSum( Byme::LogBesselI(a-1, x), Byme::LogBesselI(a+1, x) ) - log(2);
	}else{
		return Byme::LogBesselI(1, x);
	}
}

double Byme::LogVmfC(int D, double k)
{
	double halfD = (double)D * 0.5;
	return (halfD-1) * log(k) - halfD * log(2*Byme::pi) - Byme::LogBesselI(halfD-1, k);
}

double Byme::VmfA(int D, double k)
{
	double halfD = (double)D * 0.5;
	return - exp( Byme::LogBesselI(halfD, k) - Byme::LogBesselI(halfD-1, k) );
}
****************************/


//db
#include <iostream>
void Byme::expm(Ref<MatrixXd> res, const Ref<const MatrixXd> x)
{
	int n = 2;
	MatrixXd inc = x;
	res = MatrixXd::Identity(x.cols(), x.cols()) + x;
	while(inc.array().abs().maxCoeff() > 1e-18) {
		inc *= x/(n++);
		res += inc;
	}
	//db
	std::cout << "final n: " << n << std::endl;
}



////////////////////////
//                    //
//   Basic Samplers   //
//                    //
////////////////////////
#include <ctime>

/***************************************************************
 * self-implemented, based on the function rand() in <cstdlib> *
 * Invoke
 *	std::srand( unsigned(std::time(0)) );
 * in the main function before using.
 ***************************************************************/
#include <cstdlib>

double Byme::RandUnif01()
{
	return (double)std::rand() / ((double)RAND_MAX + 1.0);
}

int* Byme::RandPerm(int *rperm, int max)
{
	int i, pos;
	int swp;

	for(i=0; i<max; i++)
		rperm[i] = i;
	for(i=0; i<max-1; i++){
		pos = i + std::rand() % (max - i);
		if(pos != i){
			swp = rperm[pos];
			rperm[pos] = rperm[i];
			rperm[i] = swp;
		}
	}
	return rperm;
}
#include <memory>
int* Byme::RandPerm(int *rperm, int max, int Npart)
{
	int i, pos;
	int swp;
	int *aux;
	aux = new int[max];

	for(i=0; i<max; i++)
		aux[i] = i;
	for(i=0; i<Npart; i++){
		pos = i + std::rand() % (max - i);
		if(pos != i){
			swp = aux[pos];
			aux[pos] = aux[i];
			aux[i] = swp;
		}
	}
	memcpy( rperm, aux, Npart*sizeof(int) );
	delete [] aux;
	return rperm;
}

#include <vector> // std::vector
#include <algorithm> // std::sort
double* Byme::RandUnifSmplx(double *rv, int dim)
{
	int i;
	for(i=0; i<dim-1; i++)
		rv[i] = (double)std::rand() / ((double)RAND_MAX + 1.0);
	std::vector<double> aux(rv, rv+dim-1);
	std::sort(aux.begin(), aux.end()); // can be substituted by self-implemented quick-sort (non-recursive version), to see who is faster.
	rv[0] = aux[0];
	for(i=1; i<dim-1; i++)
		rv[i] = aux[i] - aux[i-1];
	rv[dim-1] = 1.0 - aux[dim-2];
	return rv;
}


Bystd::StdRandomEngineType STD_RANDOM_ENGINE(std::time(0));


double Bystd::RandUnif01::operator()(void)
{
	return distr(STD_RANDOM_ENGINE);
}


Bystd::RandDir::RandDir(double alpha): isScalar(true)
{
	distrs.emplace_back(alpha, 1.0);
}
Bystd::RandDir::RandDir(Ref<const VectorXd> alphas): isScalar(false)
{
	distrs.reserve(alphas.size());
	for(int i=0; i<alphas.size(); i++) distrs.emplace_back(alphas(i), 1.0);
}

void Bystd::RandDir::operator()(Ref<VectorXd> rv)
{
	double norm1 = 0;
	if(isScalar){
		for(int i=0; i<rv.size(); i++){
			rv(i) = distrs[0](STD_RANDOM_ENGINE);
			norm1 += rv(i);
		}
	}else{
		// if(rv.size() != distrs.size()) throw (int)0;
		for(int i=0; i<rv.size(); i++){
			rv(i) = distrs[i](STD_RANDOM_ENGINE);
			norm1 += rv(i);
		}
	}
	rv /= norm1;
}


double Bystd::RandNormal::operator()(void)
{
	return distr(STD_RANDOM_ENGINE);
}
Ref<MatrixXd> Bystd::RandNormal::operator()(Ref<MatrixXd> rv)
{
	for(int i=0; i<rv.rows(); i++)
		for(int j=0; j<rv.cols(); j++)
			rv(i, j) = distr(STD_RANDOM_ENGINE);
	return rv;
}

