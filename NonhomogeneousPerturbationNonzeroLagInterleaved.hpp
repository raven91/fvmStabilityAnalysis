//
// Created by Nikita Kruk on 13.10.19.
//

#ifndef FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONNONZEROLAGINTERLEAVED_HPP
#define FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONNONZEROLAGINTERLEAVED_HPP

#include "Definitions.hpp"

#include <vector>
#include <complex>
#include <unordered_map>
#include <string>
#include <atomic>

#include <eigen3/Eigen/Dense>
//#include </home/nkruk/libs/eigen3/include/eigen3/Eigen/Dense>
#include <armadillo>

class NonhomogeneousPerturbationNonzeroLagInterleaved
{
 public:

  explicit NonhomogeneousPerturbationNonzeroLagInterleaved(bool save_only_max_eigenvalues = false);
  ~NonhomogeneousPerturbationNonzeroLagInterleaved();

  void StabilityOfNontrivialSolution();

 private:

  int max_mode_idx_;
  int max_wave_number_;
  int stability_matrix_rank_;
  Eigen::MatrixXcd eigen_stability_matrix_;
  std::vector<Complex> raw_stability_matrix_;
  arma::sp_cx_dmat armadillo_stability_matrix_;
  std::vector<Complex> eigenvalues_;

  std::vector<Complex> fourier_modes_sp_; // single precision version
  std::vector<MultiprecisionComplex> fourier_modes_mp_; // multi precision version
  int max_bessel_idx_;
  std::vector<MultiprecisionReal> bessel_functions_mp_; // multi precision version
  std::unordered_map<std::string, std::vector<MultiprecisionReal>> bessel_function_table_;

  bool save_only_max_eigenvalues_;

  std::atomic<std::size_t> counter_for_bessel_functions_;

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

  void ThreeDimIdxToOneDimIdx(int n, int k_x, int k_y, int &idx);
  void OneDimIdxToThreeDimIdx(int idx, int &n, int &k_x, int &k_y);

  void StabilityOfNontrivialSolutionForParameterSet(Real v_0,
                                                    Real sigma,
                                                    Real rho,
                                                    Real alpha,
                                                    Real D_phi,
                                                    Real order_parameter,
                                                    Real velocity,
                                                    int n_wavenumber,
                                                    std::vector<Complex> &max_eigenvalue_for_fixed_wavenumber,
                                                    std::ofstream &stability_analysis_file,
                                                    std::ostringstream &output_string,
                                                    std::ostringstream &second_output_string);
  void CreateInputFilesForNontrivialSolution(std::ifstream &order_parameter_file, std::ifstream &velocity_file) const;
  void CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file,
                                             std::ofstream &second_stability_analysis_file) const;
  void InitializeBesselFunctionTable();
  bool QueryModifiedBesselFunctionsFromTable(Real sigma, Real D_phi, Real alpha);
  void CalculateModifiedBesselFunctions(const MultiprecisionReal &gamma);
  void C_1(int n,
           const MultiprecisionReal &D_phi,
           const MultiprecisionReal &velocity,
           MultiprecisionComplex &c_1) const;
  void ComputeFourierModes(Real alpha, Real D_phi, Real velocity, const MultiprecisionReal &gamma);
  void FillInStabilityMatrixForFixedWaveVector(Real v_0,
                                               Real sigma,
                                               Real rho,
                                               Real alpha,
                                               Real D_phi,
                                               Real velocity);
  void FillInStabilityMatrixForFixedPhaseMode();

};

#endif //FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONNONZEROLAGINTERLEAVED_HPP
