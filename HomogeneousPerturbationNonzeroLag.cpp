//
// Created by Nikita Kruk on 2019-06-23.
//

#include "HomogeneousPerturbationNonzeroLag.hpp"

#include <iostream>
#include <fstream>
#include <iterator> // std::ostream_iterator
#include <algorithm> // std::copy, std::fill
#include <cassert>

#include <boost/math/special_functions/bessel.hpp>

#include <eigen3/Eigen/Eigenvalues>

HomogeneousPerturbationNonzeroLag::HomogeneousPerturbationNonzeroLag(int max_mode_idx) :
    max_mode_idx_(max_mode_idx),
    stability_matrix_(Eigen::MatrixXcd::Zero(2 * max_mode_idx + 1, 2 * max_mode_idx + 1)),
    real_eigenvalues_(2 * max_mode_idx + 1, 0.0),
    eigenvalues_(2 * max_mode_idx_ + 1, Complex(0.0, 0.0)),
    fourier_modes_sp_(2 * max_mode_idx + 1, Complex(0.0, 0.0)),
    fourier_modes_mp_(2 * max_mode_idx + 1, MultiprecisionComplex(0.0, 0.0)),
    max_bessel_idx_(0),
//    bessel_functions_sp_(max_mode_idx + max_bessel_idx + 1, 0.0),
    bessel_functions_mp_(2 * max_mode_idx + 1, MultiprecisionReal(0.0))
{

}

HomogeneousPerturbationNonzeroLag::~HomogeneousPerturbationNonzeroLag()
{

}

void HomogeneousPerturbationNonzeroLag::StabilityOfUniformSolution()
{
  std::ofstream stability_analysis_file;
  CreateOutputFileForUniformSolution(stability_analysis_file);

  Complex max_eigenvalue(0.0, 0.0);

  int n_sigma = 100;
  Real sigma_min = 0.0, sigma_max = 5.0;
  Real d_sigma = (sigma_max - sigma_min) / n_sigma;
  int n_D_phi = 30;
  Real D_phi_min = 0.0, D_phi_max = 3.0;
  Real d_D_phi = (D_phi_max - D_phi_min) / n_D_phi;
  int n_alpha = 16;
  Real alpha_min = 0.0, alpha_max = 1.6;
  Real d_alpha = (alpha_max - alpha_min) / n_alpha;

//  for (int i_alpha = 0; i_alpha <= n_alpha; ++i_alpha)
  {
//    Real alpha = alpha_min + i_alpha * d_alpha;
    const Real alpha = 0.0;
    for (int i_D_phi = 1; i_D_phi <= n_D_phi; ++i_D_phi)
    {
      const Real D_phi = D_phi_min + i_D_phi * d_D_phi;
//      const Real D_phi = 0.1;
//      for (int i_sigma = 1; i_sigma <= n_sigma; ++i_sigma)
      {
//        Real sigma = sigma_min + i_sigma * d_sigma;
        const Real sigma = 4.0;

        FillInStabilityMatrixForUniformSolutions(sigma, alpha, D_phi);

        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigen_solver(stability_matrix_, false);
        Eigen::VectorXcd eigenvalues = eigen_solver.eigenvalues();
        Eigen::VectorXcd::Map(&eigenvalues_[0], eigenvalues_.size()) = eigenvalues;

        max_eigenvalue.real(std::max_element(eigenvalues_.begin(),
                                             eigenvalues_.end(),
                                             CompareComplexByRealPart())->real());
        max_eigenvalue.imag(std::max_element(eigenvalues_.begin(),
                                             eigenvalues_.end(),
                                             CompareComplexByImaginaryPart())->imag());
//        Real max_eigenvalue = *std::max_element(real_eigenvalues_.begin(), real_eigenvalues_.end());
//        std::sort(real_eigenvalues_.begin(), real_eigenvalues_.end());
        std::cout << "sigma:" << sigma << ", D_phi:" << D_phi << ", alpha:" << alpha << ", max_lambda:"
                  << max_eigenvalue << std::endl;
//        stability_analysis_file << sigma << '\t' << D_phi << '\t' << alpha << '\t' << max_eigenvalue << std::endl;

        stability_analysis_file << sigma << '\t' << D_phi << '\t' << alpha << "\t";
//        std::ostream_iterator<Real> output_iterator(stability_analysis_file, "\t");
//        std::copy(real_eigenvalues_.begin(), real_eigenvalues_.end(), output_iterator);
        for (const Complex &eigenvalue : eigenvalues_)
        {
          stability_analysis_file << eigenvalue.real() << '\t' << eigenvalue.imag() << '\t';
        } // eigenvalue
        stability_analysis_file << std::endl;
      } // i_sigma
    } // i_D_phi
  } // i_alpha

  stability_analysis_file.close();
}

void HomogeneousPerturbationNonzeroLag::CreateOutputFileForUniformSolution(std::ofstream &stability_analysis_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string folder("/home/nkruk/cpp/fvmStabilityAnalysis/output/UniformSolution/");
#elif defined(__APPLE__)
  std::string folder("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/UniformSolution/");
#endif

  stability_analysis_file.open(folder + std::string("homogeneous_perturbation_nonzero_lag.txt"),
                               std::ios::out | std::ios::trunc);
  assert(stability_analysis_file.is_open());
}

void HomogeneousPerturbationNonzeroLag::FillInStabilityMatrixForUniformSolutions(Real sigma, Real alpha, Real D_phi)
{
  Complex matrix_element = 0.0;
  for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
  {
    for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
    {
      matrix_element = 0.0;
      if (n == m)
      {
        if (n == 1)
        {
          matrix_element += 0.5 * sigma * std::exp(-kI * alpha);
        }
        if (n == -1)
        {
          matrix_element += 0.5 * sigma * std::exp(kI * alpha);
        }
        matrix_element -= D_phi * n * n;
      }

      stability_matrix_(max_mode_idx_ + n, max_mode_idx_ + m) = matrix_element;
    } // m
  } // n
}

void HomogeneousPerturbationNonzeroLag::StabilityOfNontrivialSolution()
{
  std::ifstream order_parameter_file, velocity_file;
  CreateInputFilesForNontrivialSolution(order_parameter_file, velocity_file);
  std::ofstream stability_analysis_file;
  CreateOutputFileForNontrivialSolution(stability_analysis_file);

  Real sigma = 0.0, D_phi = 0.0, alpha = 0.0, order_parameter = 0.0, velocity = 0.0;

  MultiprecisionReal gamma = 0.0;
  MultiprecisionComplex normalization_constant(0.0, 0.0);
  MultiprecisionComplex mode(0.0, 0.0), c_2(0.0, 0.0), c_2_conj(0.0, 0.0);
  MultiprecisionReal mode_real(0.0), mode_imag(0.0), c_1(0.0);
  MultiprecisionComplex exp_minus_i_n_alpha(0.0, 0.0);
  MultiprecisionComplex tmp_complex(0.0, 0.0);
  MultiprecisionReal tmp_real(0.0, 0.0);

  const MultiprecisionReal two_pi = boost::math::constants::two_pi<MultiprecisionReal>();
  const MultiprecisionComplex imaginary_unit(0.0, 1.0);

  while (order_parameter_file >> sigma >> D_phi >> alpha >> order_parameter)
  {
    velocity_file >> sigma >> D_phi >> alpha >> velocity;
    if (D_phi > 0.5 * sigma * std::cos(alpha))
    {
      continue; // the region of incoherence is not valid for the Bessel decomposition
    }
    if (!(std::isfinite(order_parameter) && std::isfinite(velocity)))
    {
      continue;
    }
    gamma = sigma * order_parameter / D_phi;

    // compute the Bessel functions of the first kind $I_\nu, \nu\in\mathbb{Z}$
    CalculateModifiedBesselFunctions(gamma);

    // compute the Fourier modes $f_n$
    std::fill(fourier_modes_mp_.begin(), fourier_modes_mp_.end(), 0.0);
    for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
    {
      c_1 = C_1(0, (MultiprecisionReal) D_phi, (MultiprecisionReal) velocity);
      c_2 = C_2(0, (MultiprecisionReal) D_phi, (MultiprecisionReal) velocity);
//      mode = bessel_functions_mp_[0] * bessel_functions_mp_[std::abs(n)] * c_1 * c_2;
      tmp_real = bessel_functions_mp_[0] * bessel_functions_mp_[std::abs(n)] * c_1;
//      Multiply(tmp_complex, c_2, tmp3);
      mode_real = tmp_real * c_2.real();
      mode_imag = tmp_real * c_2.imag();
      // function decomposition into a series of Bessel functions
      for (int nu = 1; nu <= max_bessel_idx_; ++nu)
      {
        c_1 = C_1(nu, (MultiprecisionReal) D_phi, (MultiprecisionReal) velocity);
        c_2 = C_2(nu, (MultiprecisionReal) D_phi, (MultiprecisionReal) velocity);
        c_2_conj = std::conj(c_2);
//        mode += std::pow(-1.0, Real(nu)) * bessel_functions_mp_[nu] * c_1
//            * (bessel_functions_mp_[std::abs(n + nu)] * std::conj(c_2) + bessel_functions_mp_[std::abs(n - nu)] * c_2);
//        tmp_complex = std::pow(-1.0, Real(nu)) * bessel_functions_mp_[nu] * c_1;
//        Multiply(bessel_functions_mp_[std::abs(n + nu)], std::conj(c_2), tmp2);
//        Multiply(bessel_functions_mp_[std::abs(n - nu)], c_2, tmp3);
//        Multiply(tmp_complex, tmp2 + tmp3, tmp4);
//        mode += tmp4;
        tmp_real = std::pow(-1.0, Real(nu)) * bessel_functions_mp_[nu] * c_1;
        mode_real += tmp_real * (bessel_functions_mp_[std::abs(n + nu)] * c_2_conj.real()
            + bessel_functions_mp_[std::abs(n - nu)] * c_2.real());
        mode_imag += tmp_real * (bessel_functions_mp_[std::abs(n + nu)] * c_2_conj.imag()
            + bessel_functions_mp_[std::abs(n - nu)] * c_2.imag());
      } // nu
      mode.real(mode_real);
      mode.imag(mode_imag);
//      Multiply(-imaginary_unit, n * MultiprecisionReal(alpha), tmp_complex);
//      Exp(tmp_complex, exp_minus_i_n_alpha);
      tmp_complex.real(0.0);
      tmp_complex.imag(-n * alpha);
      UnitExp(tmp_complex, exp_minus_i_n_alpha);
      Multiply(mode, exp_minus_i_n_alpha, fourier_modes_mp_[max_mode_idx_ + n]);
    } // n
//    normalization_constant = 2.0 * kMultiprecisionPi * fourier_modes_mp_[max_mode_idx_ + 0];
    Multiply(two_pi, fourier_modes_mp_[max_mode_idx_ + 0], normalization_constant);
    std::for_each(fourier_modes_mp_.begin(),
                  fourier_modes_mp_.end(),
                  [&](MultiprecisionComplex &f)
                  {
                    f /= normalization_constant.real();
//                    std::cout << f << std::endl;
                  });
    for (int n = 0; n < fourier_modes_mp_.size(); ++n)
    {
      fourier_modes_sp_[n].real((Real) fourier_modes_mp_[n].real());
      fourier_modes_sp_[n].imag((Real) fourier_modes_mp_[n].imag());
    } // n

    FillInStabilityMatrixForNontrivialSolution(sigma, alpha, D_phi);

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigen_solver(stability_matrix_, false);
    Eigen::VectorXcd eigenvalues = eigen_solver.eigenvalues();
    Eigen::VectorXd::Map(&real_eigenvalues_[0], real_eigenvalues_.size()) = eigenvalues.real();

    Real max_eigenvalue = *std::max_element(real_eigenvalues_.begin(), real_eigenvalues_.end());
    std::sort(real_eigenvalues_.begin(), real_eigenvalues_.end());
    std::cout << "sigma:" << sigma << ", D_phi:" << D_phi << ", alpha:" << alpha << ", R:" << order_parameter << ", v:"
              << velocity << ", max_lambda:" << max_eigenvalue << std::endl;

    stability_analysis_file << sigma << '\t' << D_phi << '\t' << alpha << '\t';
    std::ostream_iterator<Real> output_iterator(stability_analysis_file, "\t");
    std::copy(real_eigenvalues_.begin(), real_eigenvalues_.end(), output_iterator);
    stability_analysis_file << std::endl;
  } // read new parameters

  stability_analysis_file.close();
  velocity_file.close();
  order_parameter_file.close();
}

void HomogeneousPerturbationNonzeroLag::CreateInputFilesForNontrivialSolution(std::ifstream &order_parameter_file,
                                                                              std::ifstream &velocity_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string input_folder("/home/nkruk/cpp/fvmSelfconsistentEquations/output/HomogeneousSolutionsNonzeroLag/");
#elif defined(__APPLE__)
  std::string
      input_folder("/Users/nikita/Documents/Projects/fvm/fvmSelfconsistentEquations/HomogeneousSolutionsNonzeroLag/");
#endif

  order_parameter_file.open(input_folder + std::string("order_parameter_magnitude.txt"), std::ios::in);
  assert(order_parameter_file.is_open());
  velocity_file.open(input_folder + std::string("velocity.txt"), std::ios::in);
  assert(velocity_file.is_open());
}

void HomogeneousPerturbationNonzeroLag::CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string output_folder("/home/nkruk/cpp/fvmStabilityAnalysis/output/NontrivialSolution/");
#elif defined(__APPLE__)
  std::string output_folder("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/NontrivialSolution/");
#endif

  stability_analysis_file.open(output_folder + std::string("homogeneous_perturbation_nonzero_lag.txt"),
                               std::ios::out | std::ios::trunc);
  assert(stability_analysis_file.is_open());
}

MultiprecisionReal HomogeneousPerturbationNonzeroLag::C_1(int n,
                                                          const MultiprecisionReal &D_phi,
                                                          const MultiprecisionReal &velocity)
{
  return 1.0 / (velocity * velocity / (D_phi * D_phi) + n * n);
}

MultiprecisionComplex HomogeneousPerturbationNonzeroLag::C_2(int n,
                                                             const MultiprecisionReal &D_phi,
                                                             const MultiprecisionReal &velocity)
{
  return MultiprecisionComplex(velocity / D_phi, n);
}

void HomogeneousPerturbationNonzeroLag::CalculateModifiedBesselFunctions(const MultiprecisionReal &gamma)
{
  /*std::fill(bessel_functions_mp_.begin(), bessel_functions_mp_.end(), 0.0);
    for (int nu = 0; nu <= max_mode_idx_ + max_bessel_idx_; ++nu)
    {
      bessel_functions_mp_[nu] = boost::math::cyl_bessel_i(nu, gamma);
      std::cout << "bessel_i[" << nu << "] " << bessel_functions_mp_[nu] << std::endl;
    } // nu*/
  max_bessel_idx_ = 0;
  bessel_functions_mp_.clear();
  do
  {
//    bessel_functions_mp_.push_back(boost::math::cyl_bessel_i(max_bessel_idx_++, gamma));
    bessel_functions_mp_.emplace_back(boost::math::cyl_bessel_i(max_bessel_idx_++, gamma));
//    std::cout << "bessel_i[" << bessel_functions_mp_.size() - 1 << "] "
//              << bessel_functions_mp_[bessel_functions_mp_.size() - 1] << std::endl;
  } while ((bessel_functions_mp_.size() < 2 * max_mode_idx_ + 1)
      || (boost::multiprecision::fabs(bessel_functions_mp_[bessel_functions_mp_.size() - 1]) > 1e-50));
  max_bessel_idx_ = bessel_functions_mp_.size() - max_mode_idx_ - 1;
}

void HomogeneousPerturbationNonzeroLag::FillInStabilityMatrixForNontrivialSolution(Real sigma, Real alpha, Real D_phi)
{
  Complex matrix_element = 0.0;
  for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
  {
    for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
    {
      matrix_element = 0.0;
      if (1 == m)
      {
        matrix_element += sigma * n * M_PI * fourier_modes_sp_[max_mode_idx_ + n - 1] * std::exp(-kI * alpha);
      }
      if (n - 1 == m)
      {
        matrix_element += sigma * n * M_PI * fourier_modes_sp_[max_mode_idx_ + 1] * std::exp(-kI * alpha);
      }
      if (-1 == m)
      {
        matrix_element -= sigma * n * M_PI * fourier_modes_sp_[max_mode_idx_ + n + 1] * std::exp(kI * alpha);
      }
      if (n + 1 == m)
      {
        matrix_element -= sigma * n * M_PI * fourier_modes_sp_[max_mode_idx_ - 1] * std::exp(kI * alpha);
      }
      if (n == m)
      {
        matrix_element -= D_phi * n * n;
      }
      matrix_element += sigma * n * M_PI
          * (fourier_modes_sp_[max_mode_idx_ + n - 1] * fourier_modes_sp_[max_mode_idx_ + 1] * std::exp(-kI * alpha)
              - fourier_modes_sp_[max_mode_idx_ + n + 1] * fourier_modes_sp_[max_mode_idx_ - 1] * std::exp(kI * alpha))
          - D_phi * n * n * fourier_modes_sp_[max_mode_idx_ + n];

      stability_matrix_(max_mode_idx_ + n, max_mode_idx_ + m) = matrix_element;
    } // m
  } // n
}