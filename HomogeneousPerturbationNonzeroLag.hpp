//
// Created by Nikita Kruk on 2019-06-23.
//

#ifndef FVMSTABILITYANALYSIS_HOMOGENEOUSPERTURBATIONNONZEROLAG_HPP
#define FVMSTABILITYANALYSIS_HOMOGENEOUSPERTURBATIONNONZEROLAG_HPP

#include "Definitions.hpp"

#include <vector>
#include <complex>

#include <eigen3/Eigen/Dense>
//#include </home/nkruk/libs/eigen3/include/eigen3/Eigen/Dense>

class HomogeneousPerturbationNonzeroLag
{
 public:

  HomogeneousPerturbationNonzeroLag(int max_mode_idx = 50);
  ~HomogeneousPerturbationNonzeroLag();

  void StabilityOfUniformSolution();
  void StabilityOfNontrivialSolution();

 private:

  int max_mode_idx_;
  Eigen::MatrixXcd stability_matrix_;
  std::vector<Real> real_eigenvalues_;
  std::vector<Complex> eigenvalues_;

  std::vector<Complex> fourier_modes_sp_;
  std::vector<MultiprecisionComplex> fourier_modes_mp_;
  int max_bessel_idx_;
//  std::vector<Real> bessel_functions_sp_;
  std::vector<MultiprecisionReal> bessel_functions_mp_;

  struct CompareComplexByRealPart
  {
    inline bool operator()(const Complex &c1, const Complex &c2)
    {
      return (c1.real() < c2.real());
    }
  };
  struct CompareComplexByImaginaryPart
  {
    inline bool operator()(const Complex &c1, const Complex &c2)
    {
      return (c1.imag() < c2.imag());
    }
  };

  void CreateOutputFileForUniformSolution(std::ofstream &stability_analysis_file) const;
  void CreateInputFilesForNontrivialSolution(std::ifstream &order_parameter_file, std::ifstream &velocity_file) const;
  void CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file) const;

  MultiprecisionReal C_1(int n, const MultiprecisionReal &D_phi, const MultiprecisionReal &velocity);
  MultiprecisionComplex C_2(int n, const MultiprecisionReal &D_phi, const MultiprecisionReal &velocity);

  void FillInStabilityMatrixForUniformSolutions(Real sigma, Real alpha, Real D_phi);
  void CalculateModifiedBesselFunctions(const MultiprecisionReal &gamma);
  void FillInStabilityMatrixForNontrivialSolution(Real sigma, Real alpha, Real D_phi);

};

#endif //FVMSTABILITYANALYSIS_HOMOGENEOUSPERTURBATIONNONZEROLAG_HPP
