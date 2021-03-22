//
// Created by Nikita Kruk on 2019-06-24.
//

#ifndef FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONZEROLAG_HPP
#define FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONZEROLAG_HPP

#include "Definitions.hpp"

#include <vector>
#include <complex>
#include <atomic>

#include <eigen3/Eigen/Dense>
//#include </home/nkruk/libs/eigen3/include/eigen3/Eigen/Dense>

class NonhomogeneousPerturbationZeroLag
{
 public:

  NonhomogeneousPerturbationZeroLag(bool save_only_max_eigenvalues = false);
  ~NonhomogeneousPerturbationZeroLag();

  void StabilityOfNontrivialSolution();

 private:

  int max_mode_idx_;
  int max_wave_number_;
  std::vector<Eigen::MatrixXcd> eigen_stability_matrices_;
  std::vector<std::vector<Complex>> eigenvalues_per_thread_;

  bool save_only_max_eigenvalues_;
  std::atomic<int> counter_for_wave_numbers_;

  struct CompareComplexByRealPart
  {
    inline bool operator()(const Complex &c1, const Complex &c2)
    {
      return (c1.real() < c2.real());
    }
  };

  void CreateInputFileForNontrivialSolution(std::ifstream &order_parameter_file) const;
  void CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file) const;

  void FindMaxEigenvaluesForFixedWaveNumber(int thread_id,
                                            Real particle_velocity,
                                            Real coupling_strength,
                                            Real radius_of_interaction,
                                            Real noise,
                                            Real order_parameter,
                                            const std::vector<MultiprecisionReal> &bessel_functions,
                                            int n_wavenumber,
                                            std::vector<Complex> &max_eigenvalue_for_fixed_wavenumber,
                                            std::vector<Complex> &second_max_eigenvalue_for_fixed_wavenumber,
                                            std::ofstream &stability_analysis_file);
  void FillInStabilityMatrixForNontrivialSolution(int thread_id,
                                                  Real k_x,
                                                  Real k_y,
                                                  Real particle_velocity,
                                                  Real coupling_strength,
                                                  Real noise,
                                                  Real j_1,
                                                  const std::vector<MultiprecisionReal> &bessel_functions);
//  void FillInStabilityMatrixForNontrivialSolutionAgainstZeroWavenumber(Real coupling_strength,
//                                                                       Real noise,
//                                                                       const std::vector<MultiprecisionReal> &bessel_functions);

};

#endif //FVMSTABILITYANALYSIS_NONHOMOGENEOUSPERTURBATIONZEROLAG_HPP
