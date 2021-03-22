//
// Created by Nikita Kruk on 2019-06-23.
//

#include "HomogeneousPerturbationZeroLag.hpp"

#include <iostream>
#include <fstream>
#include <iterator> // std::ostream_iterator
#include <algorithm> // std::copy
#include <cassert>

#include <boost/math/special_functions/bessel.hpp>

#include <eigen3/Eigen/Eigenvalues>

HomogeneousPerturbationZeroLag::HomogeneousPerturbationZeroLag(int max_mode_idx) :
    max_mode_idx_(max_mode_idx),
    stability_matrix_(Eigen::MatrixXd::Zero(2 * max_mode_idx + 1, 2 * max_mode_idx + 1)),
    real_eigenvalues_(2 * max_mode_idx + 1, 0.0)
{

}

HomogeneousPerturbationZeroLag::~HomogeneousPerturbationZeroLag()
{

}

void HomogeneousPerturbationZeroLag::StabilityOfUniformSolution()
{
  std::ofstream stability_analysis_file;
  CreateOutputFileForUniformSolution(stability_analysis_file);

  int n_sigma = 100;
  Real sigma_min = 0.0, sigma_max = 5.0;
  Real d_sigma = (sigma_max - sigma_min) / n_sigma;
  int n_D_phi = 100;
  Real D_phi_min = 0.0, D_phi_max = 5.0;
  Real d_D_phi = (D_phi_max - D_phi_min) / n_D_phi;

  for (int i_D_phi = 1; i_D_phi <= n_D_phi; ++i_D_phi)
  {
    Real D_phi = D_phi_min + i_D_phi * d_D_phi;
    for (int i_sigma = 1; i_sigma <= n_sigma; ++i_sigma)
    {
      Real sigma = sigma_min + i_sigma * d_sigma;

      Real matrix_element = 0.0;
      for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
      {
        for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
        {
          matrix_element = 0.0;
          if (n == m)
          {
            if (n == 1 || n == -1)
            {
              matrix_element += 0.5 * sigma;
            }
            matrix_element -= D_phi * n * n;
          }

          stability_matrix_(max_mode_idx_ + n, max_mode_idx_ + m) = matrix_element;
        } // m
      } // n
      Eigen::ComplexEigenSolver<Eigen::MatrixXd> eigen_solver(stability_matrix_, false);
      Eigen::VectorXcd eigenvalues = eigen_solver.eigenvalues();
      Eigen::VectorXd::Map(&real_eigenvalues_[0], real_eigenvalues_.size()) = eigenvalues.real();

      Real max_eigenvalue = *std::max_element(real_eigenvalues_.begin(),
                                              real_eigenvalues_.end());
      std::cout << "sigma:" << sigma << ", D_phi:" << D_phi << ", max_lambda:" << max_eigenvalue << std::endl;
      stability_analysis_file << sigma << '\t' << D_phi << '\t' << max_eigenvalue << std::endl;
//      std::ostream_iterator<Real> output_iterator(stability_analysis_file, "\t");
//      std::copy(real_eigenvalues_.begin(), real_eigenvalues_.end(), output_iterator);
//      stability_analysis_file << std::endl;
    } // i_sigma
  } // i_D_phi

  stability_analysis_file.close();
}

void HomogeneousPerturbationZeroLag::CreateOutputFileForUniformSolution(std::ofstream &stability_analysis_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string folder("/home/nkruk/cpp/fvmStabilityAnalysis/output/UniformSolution/");
#elif defined(__APPLE__)
  std::string folder("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/UniformSolution/");
#endif

  stability_analysis_file.open(folder + std::string("homogeneous_perturbation_zero_lag.txt"),
                               std::ios::out | std::ios::trunc);
  assert(stability_analysis_file.is_open());
}

void HomogeneousPerturbationZeroLag::StabilityOfNontrivialSolution()
{
  std::ifstream order_parameter_file;
  CreateInputFileForNontrivialSolution(order_parameter_file);
  std::ofstream stability_analysis_file;
  CreateOutputFileForNontrivialSolution(stability_analysis_file);

  Real sigma = 0.0, D_phi = 0.0, order_parameter = 0.0;
  Real gamma = 0.0;
  Real I_zero = 0.0, I_one = 0.0, I_minus_one = 0.0;
  Real I_n = 0.0, I_n_minus_one = 0.0, I_n_plus_one = 0.0;
  while (order_parameter_file >> sigma >> D_phi >> order_parameter)
  {
    if (!std::isfinite(order_parameter))
    {
      continue;
    }
    gamma = sigma * order_parameter / D_phi;
    I_zero = boost::math::cyl_bessel_i(0, gamma);
    I_one = boost::math::cyl_bessel_i(1, gamma);
    I_minus_one = boost::math::cyl_bessel_i(-1, gamma);

    Real matrix_element = 0.0;
    for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
    {
      I_n = boost::math::cyl_bessel_i(n, gamma);
      I_n_minus_one = boost::math::cyl_bessel_i(n - 1, gamma);
      I_n_plus_one = boost::math::cyl_bessel_i(n + 1, gamma);
      for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
      {
        matrix_element = 0.0;
        if (1 == m)
        {
          matrix_element += 0.5 * sigma * n * I_n_minus_one / I_zero;
        }
        if (n - 1 == m)
        {
          matrix_element += 0.5 * sigma * n * I_one / I_zero;
        }
        if (-1 == m)
        {
          matrix_element -= 0.5 * sigma * n * I_n_plus_one / I_zero;
        }
        if (n + 1 == m)
        {
          matrix_element -= 0.5 * sigma * n * I_minus_one / I_zero;
        }
        if (n == m)
        {
          matrix_element -= D_phi * n * n;
        }
        if (0 == m)
        {
          matrix_element -= 0.5 * sigma * n * I_one * I_n_minus_one / (I_zero * I_zero);
          matrix_element += 0.5 * sigma * n * I_minus_one * I_n_plus_one / (I_zero * I_zero);
        }
//        matrix_element += sigma * n / (4.0 * M_PI)
//            * (I_n_minus_one * I_one / (I_zero * I_zero) - I_n_plus_one * I_minus_one / (I_zero * I_zero))
//            - D_phi * n * n * I_n / (2.0 * M_PI * I_zero);

        stability_matrix_(max_mode_idx_ + n, max_mode_idx_ + m) = matrix_element;
      } // m
    } // n

    Eigen::ComplexEigenSolver<Eigen::MatrixXd> eigen_solver(stability_matrix_, false);
    Eigen::VectorXcd eigenvalues = eigen_solver.eigenvalues();
    Eigen::VectorXd::Map(&real_eigenvalues_[0], real_eigenvalues_.size()) = eigenvalues.real();

    Real max_eigenvalue = *std::max_element(real_eigenvalues_.begin(),
                                            real_eigenvalues_.end());
    std::cout << "sigma:" << sigma << ", D_phi:" << D_phi << ", R:" << order_parameter << ", max_lambda:"
              << max_eigenvalue << std::endl;
    stability_analysis_file << sigma << '\t' << D_phi << '\t' << max_eigenvalue << std::endl;
  }

  stability_analysis_file.close();
  order_parameter_file.close();
}

void HomogeneousPerturbationZeroLag::CreateInputFileForNontrivialSolution(std::ifstream &order_parameter_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string input_folder("/home/nkruk/cpp/fvmSelfconsistentEquations/output/HomogeneousSolutionsZeroLag/");
#elif defined(__APPLE__)
  std::string
      input_folder("/Users/nikita/Documents/Projects/fvm/fvmSelfconsistentEquations/HomogeneousSolutionsZeroLag/");
#endif

  order_parameter_file.open(input_folder + std::string("order_parameter_magnitude.txt"), std::ios::in);
  assert(order_parameter_file.is_open());
}

void HomogeneousPerturbationZeroLag::CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string output_folder("/home/nkruk/cpp/fvmStabilityAnalysis/output/NontrivialSolution/");
#elif defined(__APPLE__)
  std::string output_folder("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/NontrivialSolution/");
#endif

  stability_analysis_file.open(output_folder + std::string("homogeneous_perturbation_zero_lag.txt"),
                               std::ios::out | std::ios::trunc);
  assert(stability_analysis_file.is_open());
}