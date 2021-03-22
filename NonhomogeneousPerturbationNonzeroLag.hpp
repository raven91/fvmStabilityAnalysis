//
// Created by Nikita Kruk on 2019-06-24.
//

#ifndef FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONNONZEROLAG_HPP
#define FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONNONZEROLAG_HPP

#include "Definitions.hpp"
#include "ThreadForParameterSpan.hpp"

#include <vector>
#include <complex>
#include <unordered_map>
#include <string>
#include <atomic>

#include <eigen3/Eigen/Dense>
//#include </home/nkruk/libs/eigen3/include/eigen3/Eigen/Dense>

class NonhomogeneousPerturbationNonzeroLag
{
 public:

  explicit NonhomogeneousPerturbationNonzeroLag(Thread *thread, bool save_only_max_eigenvalues = false);
  ~NonhomogeneousPerturbationNonzeroLag();

  void StabilityOfUniformSolution();
  void StabilityOfNontrivialSolution();

 private:

  Thread *thread_;

  int max_mode_idx_;
  int max_wave_number_;
  std::vector<Eigen::MatrixXcd> eigen_stability_matrices_;
  std::vector<std::vector<Complex>> raw_stability_matrices_;
  std::vector<std::vector<Complex>> eigenvalues_per_thread_;

  std::vector<Complex> fourier_modes_sp_; // single precision version
  std::vector<MultiprecisionComplex> fourier_modes_mp_; // multi precision version
  int max_bessel_idx_;
  std::vector<MultiprecisionReal> bessel_functions_mp_; // multi precision version
  std::unordered_map<std::string, std::vector<MultiprecisionReal>> bessel_function_table_;

  bool save_only_max_eigenvalues_;

  std::atomic<std::size_t> counter_for_bessel_functions_;
  std::atomic<int> counter_for_wave_numbers_;

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

  void FindMaxEigenvaluesForUniformDensity(int thread_id,
                                           Real v_0,
                                           Real sigma,
                                           Real rho,
                                           Real alpha,
                                           Real D_phi,
                                           int n_wavenumber,
                                           std::vector<Complex> &max_eigenvalue_for_fixed_wavenumber,
                                           std::ofstream &stability_analysis_file);
  void CreateOutputFileForUniformSolution(std::ofstream &stability_analysis_file) const;
  void FillInStabilityMatrixForUniformSolution(int thread_id,
                                               Real k_x,
                                               Real k_y,
                                               Real v_0,
                                               Real sigma,
                                               Real alpha,
                                               Real D_phi,
                                               Real j_1);
  void FillInStabilityMatrixForUniformSolutionAgainstZeroWavenumber(Real sigma, Real alpha, Real D_phi);

  void FindMaxEigenvaluesForNontrivialDensity(int thread_id,
                                              Real v_0,
                                              Real sigma,
                                              Real rho,
                                              Real alpha,
                                              Real D_phi,
                                              Real order_parameter,
                                              Real velocity,
                                              int n_wavenumber,
                                              std::vector<int> &number_of_unstable_modes_for_fixed_wavenumber,
                                              std::vector<Complex> &max_eigenvalue_for_fixed_wavenumber,
                                              std::vector<Complex> &second_max_eigenvalue_for_fixed_wavenumber,
                                              std::ofstream &stability_analysis_file);
  void CreateInputFilesForNontrivialSolution(std::ifstream &order_parameter_file, std::ifstream &velocity_file) const;
  void CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file,
                                             std::ofstream &second_stability_analysis_file,
                                             std::ofstream &unstable_modes_file) const;
  void InitializeBesselFunctionTable();
  bool QueryModifiedBesselFunctionsFromTable(Real sigma, Real D_phi, Real alpha);
  void CalculateModifiedBesselFunctions(const MultiprecisionReal &gamma);
  void C_1(int n,
           const MultiprecisionReal &D_phi,
           const MultiprecisionReal &velocity,
           MultiprecisionComplex &c_1) const;
  void ComputeFourierModes(Real alpha, Real D_phi, Real velocity, const MultiprecisionReal &gamma);
  void FillInStabilityMatrixForNontrivialSolution(int thread_id,
                                                  Real k_x,
                                                  Real k_y,
                                                  Real v_0,
                                                  Real sigma,
                                                  Real alpha,
                                                  Real D_phi,
                                                  Real velocity,
                                                  Real j_1);
  void FillInStabilityMatrixForNontrivialSolution(int thread_id,
                                                  Real k_x,
                                                  Real k_y,
                                                  Real v_0,
                                                  Real sigma,
                                                  Real alpha,
                                                  Real D_phi,
                                                  Real velocity,
                                                  const MultiprecisionReal &j_1);

};

#endif //FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONNONZEROLAG_HPP
