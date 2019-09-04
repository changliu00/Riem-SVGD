#pragma once

#include <cmath> // for functions: exp, log, sqrt, macros: std::isfinite, std::isnormal

///////////////////
//               //
//   Constants   //
//               //
///////////////////

// for constants: pi
// #include <boost/math/constants/constants.hpp>
// using namespace boost::math::double_constants;

namespace Byme
{

const double pi = 3.1415926535897932;

}; // Byme


///////////////////////
//                   //
//   Miscellaneous   //
//                   //
///////////////////////

namespace Byme
{

// compute log(exp(a) + exp(b))
inline double LogSum(double a, double b)
{
	if( a > b )			return a + log( 1 + exp(b-a) );
	else if( a < b )	return b + log( 1 + exp(a-b) );
	else				return a + log(2);
}

}; // Byme


namespace Byboost
{

// the inverse of the normalization constant of VMF distribution
double VmfC(int D, double k);

// the logarithm of inverse of the normalization constant of VMF distribution
double LogVmfC(int D, double k);

// the derivative of log(VmfC())
double VmfA(int D, double k);

// the derivative of bessel_i
double my_cyl_bessel_i_prime(double D, double k);

}; // Byboost


namespace Byme
{
#ifndef BymeLogBesselI_DefaultEps
#	define BymeLogBesselI_DefaultEps 1e-12
#endif
// #ifndef BymeLogBesselI_DefaultMMax
// #	define BymeLogBesselI_DefaultMMax 3000
// #endif

// basic LogBesselI's
double LogBesselISmallXEps(double a, double x, double eps = BymeLogBesselI_DefaultEps); // use given eps to stop. Eps is in the log space.
double LogBesselISmallXEpsTest(double a, double x, double eps = BymeLogBesselI_DefaultEps, bool verbose = false); // use given eps to stop. Eps is in the log space.
double LogBesselISmallXN(double a, double x, int n = 10);
double LogBesselISmallXNTest(double a, double x, int n = 10);

// basic LogBesselIBigX's
double LogBesselIBigXEps(double a, double x, double eps = BymeLogBesselI_DefaultEps); // use given eps to stop. Eps is in the original space.
double LogBesselIBigXEpsTest(double a, double x, double eps = BymeLogBesselI_DefaultEps, bool verbose = false); // use given eps to stop. Eps is in the original space.
double LogBesselIBigXN(double a, double x, int n = 10);
double LogBesselIBigXNTest(double a, double x, int n = 10);

}; // Byme


namespace Byme
{

// bessel_i acturally used
inline double LogBesselI(double a, double x)
{
	if( (a >= 50) && (a*a < 20*x) )
		return LogBesselIBigXEps(a, x);
	else
		return LogBesselISmallXEps(a, x);
}
// double LogBesselI(double a, double x); // the same as LogBesselIEps with eps being the default value. Since in most cases, LogBesselIEps is faster than boost::math::cyl_bessel_i (even when no exceptions and no infinities), so boost::math::cyl_bessel_i is discarded and the one below is deserted.
// double LogBesselI(double a, double x); // first use boost::math::cyl_bessel_i, if exception met, then return INFINITY; if -INFINITY is returned, then use LogBesselIEps with default eps. x should be small compared with a.

// the derivative of log bessel
inline double LogBesselIPrime(double a, double x)
{
	if(a != 0)
		return LogSum( Byme::LogBesselI(a-1, x), Byme::LogBesselI(a+1, x) ) - log(2);
	else
		return Byme::LogBesselI(1, x);
}

// the logarithm of inverse of the normalization constant of VMF distribution
inline double LogVmfC(int D, double k)
{
	return (0.5*D-1) * log(k) - 0.5*D * log(2 * Byme::pi) - Byme::LogBesselI(0.5*D-1, k);
}

// the derivative of log(VmfC())
inline double VmfA(int D, double k)
{
	return - exp( Byme::LogBesselI(0.5*D, k) - Byme::LogBesselI(0.5*D-1, k) );
}

}; // Byme


#include "myinclude/Eigen/Core"
using namespace Eigen;
namespace Byme
{
void expm(Ref<MatrixXd> res, const Ref<const MatrixXd> x);
}; // Byme



////////////////////////
//                    //
//   Basic Samplers   //
//                    //
////////////////////////

namespace Byme
{
/***************************************************************
 * self-implemented, based on the function rand() in <cstdlib> *
 * Invoke
 *	std::srand( unsigned(std::time(0)) );
 * in the main function before using.
 ***************************************************************/

// Uniformly distributed random variable on the interval [0, 1)
double RandUnif01();

// random permutation of 0, 1, ..., max-1
int* RandPerm(int *rperm, int max);
// Only the first Npart munbers of the random permutation is returned. 
// Can also be viewed as random Npart selections from 0, 1, ..., max-1.
int* RandPerm(int *rperm, int max, int Npart);

// randomly sample from the (dim-1)-dimensional simplex, expressed in the dim-dimensional Euclidean space
double* RandUnifSmplx(double *rv, int dim);

}; // Byme

#include <vector>
#include <random> // calls for c++11
#include "myinclude/Eigen/Core"
using namespace Eigen;
namespace Bystd
{

typedef std::mt19937_64 StdRandomEngineType;

class RandUnif01
{
public:
	std::uniform_real_distribution<double> distr;

public:
	explicit RandUnif01(): distr(0.0, 1.0) {};
	~RandUnif01(){};
	double operator()(void);
};

class RandDir
{
public:
	const bool isScalar;
	std::vector< std::gamma_distribution<double> > distrs;

public:
	explicit RandDir(double alpha);
	explicit RandDir(Ref<const VectorXd> alphas);
	void operator()(Ref<VectorXd> rv);
};

class RandNormal
{
public:
	std::normal_distribution<double> distr;

public:
	explicit RandNormal(): distr(0.0, 1.0) {};
	double operator()(void);
	Ref<MatrixXd> operator()(Ref<MatrixXd> rv);
};

}; // Bystd

