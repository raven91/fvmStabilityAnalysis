//
// Created by Nikita Kruk on 2019-06-23.
//

#ifndef FVMSTABILITYANALYSIS_DEFINITIONS_HPP
#define FVMSTABILITYANALYSIS_DEFINITIONS_HPP

//#define BCS_CLUSTER
#define EIGEN_USE_MKL_ALL
//#define EIGEN_USE_BLAS

#include <complex>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/math/constants/constants.hpp>

typedef double Real;
typedef std::complex<double> Complex;
#define MKL_Complex16 Complex
//typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<500>> MultiprecisionReal;
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<0>> MultiprecisionReal;
//typedef boost::multiprecision::mpfr_float_1000 MultiprecisionReal;
typedef std::complex<MultiprecisionReal> MultiprecisionComplex;

const Real kPi = boost::math::constants::pi<Real>();
const Real kTwoPi = boost::math::constants::two_pi<Real>();
const Complex kI = Complex(0.0, 1.0);

extern int kRank; // the rank of a process when MPI does not work with multiprecision
extern int kBesselThreads; // number of std::threads to compute the modified Bessel functions
extern int kWaveNumberThreads; // number of std::threads to loop through the wave numbers k\in\mathbb{Z}^2

// Additional operators for the multiprecision operations
/*[[deprecated("Excessive copying renders slower performance")]]
MultiprecisionComplex operator*(const MultiprecisionReal &lhs, const MultiprecisionComplex &rhs);
[[deprecated("Excessive copying renders slower performance")]]
MultiprecisionComplex operator*(const MultiprecisionComplex &lhs, const MultiprecisionReal &rhs);
[[deprecated("Excessive copying renders slower performance")]]
MultiprecisionComplex operator*(const MultiprecisionComplex &lhs, const MultiprecisionComplex &rhs);
[[deprecated("Excessive copying renders slower performance")]]
MultiprecisionComplex Exp(const MultiprecisionComplex &x);*/

void Multiply(const MultiprecisionReal &lhs, const MultiprecisionComplex &rhs, MultiprecisionComplex &res);
void Multiply(const MultiprecisionComplex &lhs, const MultiprecisionReal &rhs, MultiprecisionComplex &res);
void Multiply(const MultiprecisionComplex &lhs, const MultiprecisionComplex &rhs, MultiprecisionComplex &res);
void Exp(const MultiprecisionComplex &x, MultiprecisionComplex &res);
void UnitExp(const MultiprecisionComplex &x, MultiprecisionComplex &res);

#endif //FVMSTABILITYANALYSIS_DEFINITIONS_HPP
