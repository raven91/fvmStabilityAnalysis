//
// Created by Nikita Kruk on 2019-07-05.
//

#include "Definitions.hpp"

/*MultiprecisionComplex operator*(const MultiprecisionReal &lhs, const MultiprecisionComplex &rhs)
{
  return MultiprecisionComplex(lhs * rhs.real(), lhs * rhs.imag());
}

MultiprecisionComplex operator*(const MultiprecisionComplex &lhs, const MultiprecisionReal &rhs)
{
  return MultiprecisionComplex(lhs.real() * rhs, lhs.imag() * rhs);
}

MultiprecisionComplex operator*(const MultiprecisionComplex &lhs, const MultiprecisionComplex &rhs)
{
  return MultiprecisionComplex(lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
                               lhs.real() * rhs.imag() + lhs.imag() * rhs.real());
}

MultiprecisionComplex Exp(const MultiprecisionComplex &x)
{
  MultiprecisionReal radius(exp(x.real()));
  return MultiprecisionComplex(radius * cos(x.imag()), radius * sin(x.imag()));
}*/

void Multiply(const MultiprecisionReal &lhs, const MultiprecisionComplex &rhs, MultiprecisionComplex &res)
{
  res.real(lhs * rhs.real());
  res.imag(lhs * rhs.imag());
}

void Multiply(const MultiprecisionComplex &lhs, const MultiprecisionReal &rhs, MultiprecisionComplex &res)
{
  res.real(lhs.real() * rhs);
  res.imag(lhs.imag() * rhs);
}

void Multiply(const MultiprecisionComplex &lhs, const MultiprecisionComplex &rhs, MultiprecisionComplex &res)
{
  res.real(lhs.real() * rhs.real() - lhs.imag() * rhs.imag());
  res.imag(lhs.real() * rhs.imag() + lhs.imag() * rhs.real());
}

void Exp(const MultiprecisionComplex &x, MultiprecisionComplex &res)
{
  MultiprecisionReal radius(boost::multiprecision::exp(x.real()));
  res.real(radius * boost::multiprecision::cos(x.imag()));
  res.imag(radius * boost::multiprecision::sin(x.imag()));
}

void UnitExp(const MultiprecisionComplex &x, MultiprecisionComplex &res)
{
  res.real(boost::multiprecision::cos(x.imag()));
  res.imag(boost::multiprecision::sin(x.imag()));
}