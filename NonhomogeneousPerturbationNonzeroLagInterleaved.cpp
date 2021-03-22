//
// Created by Nikita Kruk on 13.10.19.
//

#include "NonhomogeneousPerturbationNonzeroLagInterleaved.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator> // std::ostream_iterator, std::copy
#include <algorithm> // std::copy, std::fill
#include <cassert>
#include <chrono>
#include <thread>
#include <mutex>

#include <boost/math/special_functions/bessel.hpp>

#include <mkl.h>
#include <eigen3/Eigen/Eigenvalues>
//#include </home/nkruk/libs/eigen3/include/eigen3/Eigen/Eigenvalues>

NonhomogeneousPerturbationNonzeroLagInterleaved::NonhomogeneousPerturbationNonzeroLagInterleaved(bool save_only_max_eigenvalues)
    :
    max_mode_idx_(100),
    max_wave_number_(3),
    stability_matrix_rank_((2 * max_mode_idx_ + 1) * (2 * max_wave_number_ + 1) * (max_wave_number_ + 1)),
    eigen_stability_matrix_(),//(Eigen::MatrixXcd::Zero(stability_matrix_rank_, stability_matrix_rank_)),
    raw_stability_matrix_(),//(stability_matrix_rank_ * stability_matrix_rank_, Complex(0.0, 0.0)),
    armadillo_stability_matrix_(stability_matrix_rank_, stability_matrix_rank_),
    eigenvalues_(stability_matrix_rank_, Complex(0.0, 0.0)),
    fourier_modes_sp_(2 * max_mode_idx_ + 1, Complex(0.0, 0.0)),
    fourier_modes_mp_(2 * max_mode_idx_ + 1, MultiprecisionComplex(0.0, 0.0)),
    max_bessel_idx_(700),
    bessel_functions_mp_(max_mode_idx_ + max_bessel_idx_ + 1, MultiprecisionReal(0.0)),
    save_only_max_eigenvalues_(save_only_max_eigenvalues),
    bessel_function_table_(),
    counter_for_bessel_functions_(0)
{

}

NonhomogeneousPerturbationNonzeroLagInterleaved::~NonhomogeneousPerturbationNonzeroLagInterleaved() = default;

void NonhomogeneousPerturbationNonzeroLagInterleaved::ThreeDimIdxToOneDimIdx(int n, int k_x, int k_y, int &idx)
{
  // the winding order is n->k_x->k_y
  idx = n + (2 * max_mode_idx_ + 1) * (k_x + (2 * max_wave_number_ + 1) * k_y);
}

/**
 *
 * @param idx
 * @param n in [0, 2*max_mode_idx_+1)
 * @param k_x in [0, 2*max_wave_number+1)
 * @param k_y in [0, max_wave_number+1)
 */
void NonhomogeneousPerturbationNonzeroLagInterleaved::OneDimIdxToThreeDimIdx(int idx, int &n, int &k_x, int &k_y)
{
  // the winding order is n->k_x->k_y
  k_y = idx / ((2 * max_mode_idx_ + 1) * (2 * max_wave_number_ + 1));
  k_x = (idx % ((2 * max_mode_idx_ + 1) * (2 * max_wave_number_ + 1))) / (2 * max_mode_idx_ + 1);
  n = idx % (2 * max_mode_idx_ + 1);
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::StabilityOfNontrivialSolution()
{
  std::ifstream order_parameter_file, velocity_file;
  CreateInputFilesForNontrivialSolution(order_parameter_file, velocity_file);
  std::ofstream stability_analysis_file, second_stability_analysis_file;
  CreateOutputFileForNontrivialSolution(stability_analysis_file, second_stability_analysis_file);
//  InitializeBesselFunctionTable();

  const int n_wavenumber = (2 * max_wave_number_ + 1) * (max_wave_number_ + 1);
  std::vector<Complex> max_eigenvalue_for_fixed_wavenumber(n_wavenumber, Complex(0.0, 0.0));
  Complex max_max_eigenvalue(0.0, 0.0);

  Real scale = 100.0;
  Real sigma(0.0), D_phi(0.0), alpha(0.0), order_parameter(0.0), velocity(0.0);
  std::vector<Real> rhos{0.01};//{-1.0, 0.01, 0.1, 0.25, 0.5};
  std::vector<Real> v_0s{0.01};//{0.001, 0.01, 0.1, 1.0};
  /*if (!save_only_max_eigenvalues_)
  {
    std::cout << "rho:>";
    std::cin >> rho;
    rhos = std::vector<Real>(1, rho);
  }*/
  /*for (int i = 0; i < 10; ++i)
  {
    rhos.push_back((i + 1) * 0.05 / scale);
  }*/
  MultiprecisionReal gamma = 0.0;

  std::chrono::time_point<std::chrono::system_clock> timer = std::chrono::system_clock::now();

  /*std::ifstream input_parameter_file
      ("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/NontrivialSolution/SelectedParameterValues/selected_parameter_values.txt", std::ios::in);
  for (int i_rho = 1; i_rho <= 10; ++i_rho)
  {
    rhos.push_back(i_rho * 0.05);
  } // i_rho*/
//  sigma = 1.0; D_phi = 0.0025; alpha = 1.48; order_parameter = 0.871613468980922; velocity = -0.867275136031329;
//  while (input_parameter_file >> sigma >> D_phi >> alpha >> v_0s[0] >> rhos[0] >> order_parameter >> velocity)
  while (order_parameter_file >> sigma >> D_phi >> alpha >> order_parameter)
  {
    velocity_file >> sigma >> D_phi >> alpha >> velocity;
    if ((D_phi > 0.5 * sigma * std::cos(alpha)) // the region of incoherence is not valid for the Bessel decomposition
        || !(std::isfinite(order_parameter) && std::isfinite(velocity) && std::isfinite(1.0 / velocity)))
    {
      continue;
    }
    gamma = sigma * order_parameter / D_phi;

    // compute the modified Bessel functions of the first kind $I_\nu, \nu\in\mathbb{Z}$
    if (!QueryModifiedBesselFunctionsFromTable(sigma, D_phi, alpha))
    {
//      std::cout << "manual calculation\n";
//      CalculateModifiedBesselFunctions(gamma);
      counter_for_bessel_functions_ = 0;
      std::vector<std::thread> bessel_threads(kBesselThreads);
      for (int i = 0; i < kBesselThreads; ++i)
      {
        bessel_threads[i] =
            std::thread(&NonhomogeneousPerturbationNonzeroLagInterleaved::CalculateModifiedBesselFunctions,
                        this,
                        std::cref(gamma));
      } // i
      for (auto &bessel_thread : bessel_threads)
      {
        bessel_thread.join();
      } // bessel_thread
    } // go to the Bessel function of the next order

    // compute the Fourier modes $f_n$, complex conjugates are not computed explicitly here
    ComputeFourierModes(alpha, D_phi, velocity, gamma);

    for (const Real v_0 : v_0s)
    {
      for (const Real rho : rhos)
      {
        std::ostringstream output_string, second_output_string;
        StabilityOfNontrivialSolutionForParameterSet(v_0,
                                                     sigma,
                                                     rho,
                                                     alpha,
                                                     D_phi,
                                                     order_parameter,
                                                     velocity,
                                                     n_wavenumber,
                                                     max_eigenvalue_for_fixed_wavenumber,
                                                     stability_analysis_file,
                                                     output_string,
                                                     second_output_string);
        if (save_only_max_eigenvalues_)
        {
          max_max_eigenvalue = *std::max_element(&max_eigenvalue_for_fixed_wavenumber[0],
                                                 &max_eigenvalue_for_fixed_wavenumber[n_wavenumber + 1],
                                                 CompareComplexByRealPart());
          std::cout << "[" << kRank << "]\t" << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi
                    << '\t' << max_max_eigenvalue << std::endl;
          stability_analysis_file << output_string.str() << std::endl;
          second_stability_analysis_file << second_output_string.str() << std::endl;
          /*output_parameter_file << output_string.str() << std::endl;*/
        }
      } // rho
    } // v_0
    std::chrono::duration<Real> elapsed_seconds = std::chrono::system_clock::now() - timer;
    std::cout << "time per cycle: " << elapsed_seconds.count() << "s" << std::endl;
    timer = std::chrono::system_clock::now();
  } // read new parameters

  stability_analysis_file.close();
  velocity_file.close();
  order_parameter_file.close();
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::StabilityOfNontrivialSolutionForParameterSet(Real v_0,
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
                                                                                                   std::ostringstream &second_output_string)
{
//  const Real wavenumber_min = 0.0, wavenumber_max = 10.0;
//  const Real d_wavenumber = (wavenumber_max - wavenumber_min) / n_wavenumber;
//  const int n_max_wavenumber_to_save = 1;

  Complex max_eigenvalue(0.0, 0.0), second_max_eigenvalue(0.0, 0.0);

  std::fill(max_eigenvalue_for_fixed_wavenumber.begin(),
            max_eigenvalue_for_fixed_wavenumber.end(),
            Complex(0.0, 0.0));
  output_string << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi << '\t';
  second_output_string << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi << '\t';
  std::fill(raw_stability_matrix_.begin(), raw_stability_matrix_.end(), Complex(0.0, 0.0));
  std::fill(armadillo_stability_matrix_.begin(), armadillo_stability_matrix_.end(), Complex(0.0, 0.0));
  FillInStabilityMatrixForFixedWaveVector(v_0, sigma, rho, alpha, D_phi, velocity);
  FillInStabilityMatrixForFixedPhaseMode();

  /*eigen_stability_matrix_ =
          Eigen::Map<Eigen::MatrixXcd>(raw_stability_matrix_.data(), stability_matrix_rank_, stability_matrix_rank_);
      eigen_stability_matrix_.transposeInPlace();
      Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigen_solver(eigen_stability_matrix_, false);
      const Eigen::VectorXcd &eigenvalues = eigen_solver.eigenvalues();
      Eigen::VectorXcd::Map(&eigenvalues_[0], eigenvalues_.size()) = eigenvalues;*/
  /*int sdim = 0;
  LAPACKE_zgees(LAPACK_ROW_MAJOR,
                'N',
                'N',
                nullptr,
                stability_matrix_rank_,
                &raw_stability_matrix_[0],
                stability_matrix_rank_,
                &sdim,
                &eigenvalues_[0],
                nullptr,
                stability_matrix_rank_);*/
  arma::cx_vec eigval;
  arma::cx_mat eigvec;
  std::cout << "stability matrix size: " << armadillo_stability_matrix_.size() << ", "
            << "nnz: " << armadillo_stability_matrix_.n_nonzero << std::endl;
//  for (auto it = armadillo_stability_matrix_.begin(); it != armadillo_stability_matrix_.end(); ++it)
//  {
//    std::cout << "[" << it.row() << "," << it.col() << "] " << *it << std::endl;
//  }
  arma::eigs_gen(eigval, eigvec, armadillo_stability_matrix_, 10, "lr");

  for (int k_x = 0; k_x < 2 * max_wave_number_ + 1; ++k_x)
  {
    for (int k_y = 0; k_y < max_wave_number_ + 1; ++k_y)
    {
//      int idx_first = 0, idx_last = 0;
//      ThreeDimIdxToOneDimIdx(0, k_x, k_y, idx_first);
//      ThreeDimIdxToOneDimIdx(2 * max_mode_idx_ + 1, k_x, k_y, idx_last);
//      max_eigenvalue = *std::max_element(&eigenvalues_[idx_first], &eigenvalues_[idx_last], CompareComplexByRealPart());
      max_eigenvalue = *std::max_element(eigval.begin(), eigval.end(), CompareComplexByRealPart());
      max_eigenvalue_for_fixed_wavenumber[k_x + (2 * max_wave_number_ + 1) * k_y] = max_eigenvalue;

      if (!save_only_max_eigenvalues_)
      {
        std::cout << "v_0:" << v_0 << ", sigma:" << sigma << ", rho:" << rho << ", D_phi:" << D_phi << ", alpha:"
                  << alpha << ", R:" << order_parameter << ", v:" << velocity << ", wavenumber:("
                  << k_x - max_wave_number_ << "," << k_y << "), max_lambda:" << max_eigenvalue << std::endl;
        std::ostringstream local_output_string;
        local_output_string << k_x << '\t' << k_y << '\t';
//        for (int idx = idx_first; idx < idx_last; ++idx)
//        {
//          local_output_string << eigenvalues_[idx].real() << '\t' << eigenvalues_[idx].imag() << '\t';
//        } // eigenvalue
        local_output_string << eigval[0].real() << '\t' << eigval[0].imag() << '\t';
        stability_analysis_file << local_output_string.str() << std::endl;
      } else
      {
        output_string << max_eigenvalue.real() << '\t' << max_eigenvalue.imag() << '\t';
      }
    } // k_y
  } // k_x
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::CreateInputFilesForNontrivialSolution(std::ifstream &order_parameter_file,
                                                                                            std::ifstream &velocity_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string input_folder("/home/nkruk/cpp/fvmSelfconsistentEquations/output/HomogeneousSolutionsNonzeroLag/");
#elif defined(__APPLE__)
  std::string
      input_folder("/Users/nikita/Documents/Projects/fvm/fvmSelfconsistentEquations/HomogeneousSolutionsNonzeroLag/");
#endif

#if defined(BCS_CLUSTER)
  order_parameter_file.open(
      input_folder + std::string("order_parameter_magnitude_nq") + std::to_string(kRank) + std::string(".txt"),
      std::ios::in);
  velocity_file.open(input_folder + std::string("velocity_nq") + std::to_string(kRank) + std::string(".txt"),
                     std::ios::in);
#else
  order_parameter_file.open(input_folder + std::string("order_parameter_magnitude.txt"), std::ios::in);
  velocity_file.open(input_folder + std::string("velocity.txt"), std::ios::in);
#endif
  assert(order_parameter_file.is_open());
  assert(velocity_file.is_open());
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file,
                                                                                            std::ofstream &second_stability_analysis_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string output_folder("/home/nkruk/cpp/fvmStabilityAnalysis/output/NontrivialSolution/");
#elif defined(__APPLE__)
  std::string output_folder("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/NontrivialSolution/");
#endif

#if defined(BCS_CLUSTER)
  stability_analysis_file.open(
      output_folder + std::string("nonhomogeneous_perturbation_nonzero_lag_nq") + std::to_string(kRank)
          + std::string(".txt"), std::ios::out | std::ios::trunc);
  second_stability_analysis_file.open(
      output_folder + std::string("second_nonhomogeneous_perturbation_nonzero_lag_nq") + std::to_string(kRank)
          + std::string(".txt"), std::ios::out | std::ios::trunc);
#else
  stability_analysis_file.open(output_folder + std::string("nonhomogeneous_perturbation_nonzero_lag.txt"),
                               std::ios::out | std::ios::trunc);
  second_stability_analysis_file.open(output_folder + std::string("second_nonhomogeneous_perturbation_nonzero_lag.txt"),
                                      std::ios::out | std::ios::trunc);
#endif
  assert(stability_analysis_file.is_open());
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::C_1(int n,
                                                          const MultiprecisionReal &D_phi,
                                                          const MultiprecisionReal &velocity,
                                                          MultiprecisionComplex &c_1) const
{
  static MultiprecisionReal modulus(0.0);
  modulus = (velocity * velocity) / (D_phi * D_phi) + n * n;
  c_1.real(velocity / D_phi / modulus);
  c_1.imag(n / modulus);
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::InitializeBesselFunctionTable()
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string input_folder("/home/nkruk/cpp/fvmBesselFunctionTable/output/");
#elif defined(__APPLE__)
  std::string
      input_folder("/Users/nikita/Documents/Projects/fvm/fvmBesselFunctionTable/");
#endif

  std::ifstream bessel_function_table_file;
#if defined(BCS_CLUSTER)
  bessel_function_table_file.open(
      input_folder + std::string("bessel_function_table_s") + std::to_string(kRank) + std::string(".txt"),
      std::ios::in);
#else
  bessel_function_table_file.open(input_folder + std::string("bessel_function_table.txt"), std::ios::in);
#endif
  assert(bessel_function_table_file.is_open());

  std::string table_line;
  while (std::getline(bessel_function_table_file, table_line))
  {
    std::istringstream table_line_stream(table_line);
    std::string sigma, D_phi, alpha;
    table_line_stream >> sigma >> D_phi >> alpha;

    std::string key = sigma + std::string("|") + D_phi + std::string("|") + alpha;
    bessel_function_table_[key].reserve(max_mode_idx_ + max_bessel_idx_ + 1);

    std::string bessel_i;
    for (int nu = 0; nu <= max_mode_idx_ + max_bessel_idx_; ++nu)
    {
      table_line_stream >> bessel_i;
      bessel_function_table_[key].emplace_back(bessel_i);
    } // nu
//    while (table_line_stream >> bessel_i)
//    {
//      bessel_function_table_[key].emplace_back(bessel_i);
//    }
  }
}

bool NonhomogeneousPerturbationNonzeroLagInterleaved::QueryModifiedBesselFunctionsFromTable(Real sigma,
                                                                                            Real D_phi,
                                                                                            Real alpha)
{
  std::ostringstream key;
  key << sigma << "|" << D_phi << "|" << alpha;
  if (bessel_function_table_.find(key.str()) != bessel_function_table_.end())
  {
    bessel_functions_mp_ = bessel_function_table_[key.str()];
    return true;
  } else
  {
    return false;
  }
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::CalculateModifiedBesselFunctions(const MultiprecisionReal &gamma)
{
  /*std::fill(bessel_functions_mp_.begin(), bessel_functions_mp_.end(), 0.0);
    for (int nu = 0; nu <= max_mode_idx_ + max_bessel_idx_; ++nu)
    {
      bessel_functions_mp_[nu] = boost::math::cyl_bessel_i(nu, gamma);
      std::cout << "bessel_i[" << nu << "] " << bessel_functions_mp_[nu] << std::endl;
    } // nu*/
  /*max_bessel_idx_ = 0;
  bessel_functions_mp_.clear();
  do
  {
//    bessel_functions_mp_.push_back(boost::math::cyl_bessel_i(max_bessel_idx_++, gamma));
    bessel_functions_mp_.emplace_back(boost::math::cyl_bessel_i(max_bessel_idx_++, gamma));
    std::cout << "bessel_i[" << bessel_functions_mp_.size() - 1 << "] "
              << bessel_functions_mp_[bessel_functions_mp_.size() - 1] << std::endl;
  } while ((bessel_functions_mp_.size() < 2 * max_mode_idx_ + 1)
      || (boost::multiprecision::fabs(bessel_functions_mp_[bessel_functions_mp_.size() - 1]) > 1e-20));
  max_bessel_idx_ = bessel_functions_mp_.size() - max_mode_idx_ - 1;*/

//  for (std::size_t nu = 0; nu < bessel_functions_mp_.size(); ++nu)
//  {
//    bessel_functions_mp_[nu] = boost::math::cyl_bessel_i(nu, gamma);
////    std::cout << "bessel_i[" << nu << "] " << bessel_functions_mp_[nu] << std::endl;
//  } // nu

  std::size_t nu = std::atomic_fetch_add(&counter_for_bessel_functions_, std::size_t(1));
  while (nu < bessel_functions_mp_.size())
  {
    bessel_functions_mp_[nu] = boost::math::cyl_bessel_i(nu, gamma);
    nu = std::atomic_fetch_add(&counter_for_bessel_functions_, std::size_t(1));
  }
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::ComputeFourierModes(Real alpha,
                                                                          Real D_phi,
                                                                          Real velocity,
                                                                          const MultiprecisionReal &gamma)
{
  static MultiprecisionReal m_D_phi(0.0), m_velocity(0.0);
  static MultiprecisionComplex normalization_constant(0.0, 0.0);
  static MultiprecisionComplex mode(0.0, 0.0), c_1(0.0, 0.0);//, c_2(0.0, 0.0), c_2_conj(0.0, 0.0);
  static MultiprecisionReal mode_real(0.0), mode_imag(0.0);
  static MultiprecisionComplex exp_minus_i_n_alpha(0.0, 0.0);
  static MultiprecisionComplex tmp_complex(0.0, 0.0);
  static MultiprecisionReal tmp_real(0.0);

  static const MultiprecisionReal two_pi = boost::math::constants::two_pi<MultiprecisionReal>();
  m_D_phi = (MultiprecisionReal) D_phi;
  m_velocity = (MultiprecisionReal) velocity;

//  std::fill(fourier_modes_mp_.begin(), fourier_modes_mp_.end(), 0.0);
  for (int n = 0; n <= max_mode_idx_; ++n)
  {
    C_1(n, m_D_phi, m_velocity, c_1);
    tmp_real = bessel_functions_mp_[0] * std::pow(-1.0, 1.0 * n) * bessel_functions_mp_[std::abs(n)];// * c_1;
    mode_real = tmp_real * c_1.real();
    mode_imag = tmp_real * c_1.imag();
    // function decomposition into a series of Bessel functions
    for (int nu = 1; nu <= max_bessel_idx_; ++nu)
    {
      C_1(n + nu, m_D_phi, m_velocity, c_1);
      tmp_real = bessel_functions_mp_[nu] * std::pow(-1.0, 1.0 * (n + nu)) * bessel_functions_mp_[std::abs(n + nu)];
      mode_real += tmp_real * c_1.real();
      mode_imag += tmp_real * c_1.imag();

      C_1(n - nu, m_D_phi, m_velocity, c_1);
      tmp_real = bessel_functions_mp_[nu] * std::pow(-1.0, 1.0 * (n - nu)) * bessel_functions_mp_[std::abs(n - nu)];
      mode_real += tmp_real * c_1.real();
      mode_imag += tmp_real * c_1.imag();
    } // nu
    mode.real(mode_real);
    mode.imag(mode_imag);

    tmp_complex.real(0.0);
    tmp_complex.imag(n * (velocity * 0.0 - alpha));
    UnitExp(tmp_complex, exp_minus_i_n_alpha);

    Multiply(mode, exp_minus_i_n_alpha, fourier_modes_mp_[max_mode_idx_ + n]);
  } // n
  Multiply(two_pi, fourier_modes_mp_[max_mode_idx_ + 0], normalization_constant);
//  int i = 0;
  std::for_each(&fourier_modes_mp_[max_mode_idx_ + 0],
                &fourier_modes_mp_[2 * max_mode_idx_ + 1],
                [&](MultiprecisionComplex &f)
                {
                  f /= normalization_constant.real();
//                  std::cout << "f[" << i++ << "] " << f << '\t';
                });
//  std::cout << std::endl;

  // negative Fourier coefficients are complex conjugates of the positive ones
  for (int n = 0; n <= max_mode_idx_; ++n)
  {
    fourier_modes_sp_[max_mode_idx_ + n].real(fourier_modes_mp_[max_mode_idx_
        + n].real().template convert_to<Real>());
    fourier_modes_sp_[max_mode_idx_ + n].imag(fourier_modes_mp_[max_mode_idx_
        + n].imag().template convert_to<Real>());

    fourier_modes_sp_[max_mode_idx_ - n] = std::conj(fourier_modes_sp_[max_mode_idx_ + n]);
  } // n
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::FillInStabilityMatrixForFixedWaveVector(Real v_0,
                                                                                              Real sigma,
                                                                                              Real rho,
                                                                                              Real alpha,
                                                                                              Real D_phi,
                                                                                              Real velocity)
{
  const Complex &f_0 = fourier_modes_sp_[max_mode_idx_ + 0];
  const Complex f_0_inverse = 1.0 / f_0;
  const Complex &f_1 = fourier_modes_sp_[max_mode_idx_ + 1];
  const Complex &f_minus_1 = fourier_modes_sp_[max_mode_idx_ - 1];
  const Complex exp_plus_i_alpha = std::exp(kI * alpha), exp_minus_i_alpha = std::exp(-kI * alpha);

  Real j_1 = 0.0, k_abs = 0.0;
  Complex matrix_element(0.0, 0.0);
//  int nn = 0, mm = 0;
  int n = 0, k_x = 0, k_y = 0;
  int m = 0, q_x = 0, q_y = 0;
  for (int idx_1 = 0; idx_1 < stability_matrix_rank_; ++idx_1)
  {
    OneDimIdxToThreeDimIdx(idx_1, n, k_x, k_y);
    n -= max_mode_idx_;
    k_x -= max_wave_number_;
    if (((k_x == 0) && (k_y == 0)) || (rho < 0.0))
    {
      j_1 = 1.0;
    } else
    {
      k_abs = std::sqrt(k_x * k_x + k_y * k_y);
      j_1 = 2.0 * boost::math::cyl_bessel_j(1, 2.0 * kPi * rho * k_abs) / (2.0 * kPi * rho * k_abs);
    }

    for (int idx_2 = 0; idx_2 < stability_matrix_rank_; ++idx_2)
    {
      OneDimIdxToThreeDimIdx(idx_2, m, q_x, q_y);
      m -= max_mode_idx_;
      q_x -= max_wave_number_;
      matrix_element.real(0.0), matrix_element.imag(0.0);

      bool is_nonzero = false;
      if ((k_x == q_x) && (k_y == q_y))
      {
        if (n - 1 == m)
        {
          matrix_element += 0.5 * v_0 * 2.0 * kPi * (kI * Real(k_x) - Real(k_y));
          matrix_element += 0.5 * n * sigma * f_0_inverse * f_1 * exp_minus_i_alpha;
          is_nonzero = true;
        }
        if (n + 1 == m)
        {
          matrix_element += 0.5 * v_0 * 2.0 * kPi * (kI * Real(k_x) + Real(k_y));
          matrix_element -= 0.5 * n * sigma * f_0_inverse * f_minus_1 * exp_plus_i_alpha;
          is_nonzero = true;
        }
        if (n == m)
        {
          matrix_element -= (n * n * D_phi + kI * Real(n) * velocity) + kI * velocity * kPi * Real(k_x - k_y);
          is_nonzero = true;
        }
        if (1 == m)
        {
          matrix_element +=
              0.5 * n * sigma * f_0_inverse * j_1 * fourier_modes_sp_[max_mode_idx_ + n - 1] * exp_minus_i_alpha;
          is_nonzero = true;
        }
        if (-1 == m)
        {
          matrix_element -=
              0.5 * n * sigma * f_0_inverse * j_1 * fourier_modes_sp_[max_mode_idx_ + n + 1] * exp_plus_i_alpha;
          is_nonzero = true;
        }
        if (0 == m)
        {
          matrix_element -=
              0.5 * n * sigma * f_0_inverse * j_1 * f_1 * fourier_modes_sp_[max_mode_idx_ + n - 1] * f_0_inverse
                  * exp_minus_i_alpha;
          matrix_element +=
              0.5 * n * sigma * f_0_inverse * j_1 * f_minus_1 * fourier_modes_sp_[max_mode_idx_ + n + 1] * f_0_inverse
                  * exp_plus_i_alpha;
          is_nonzero = true;
        }
      }

      // raw major order
//      raw_stability_matrix_[idx_2 + idx_1 * stability_matrix_rank_] = matrix_element;
      if (is_nonzero)
      {
        armadillo_stability_matrix_(idx_1, idx_2) = matrix_element;
      }
    } // idx_2
  } // idx_1
}

void NonhomogeneousPerturbationNonzeroLagInterleaved::FillInStabilityMatrixForFixedPhaseMode()
{
  Complex matrix_element(0.0, 0.0);
  int n = 0, k_x = 0, k_y = 0;
  int m = 0, q_x = 0, q_y = 0;
  for (int idx_1 = 0; idx_1 < stability_matrix_rank_; ++idx_1)
  {
    OneDimIdxToThreeDimIdx(idx_1, n, k_x, k_y);
    n -= max_mode_idx_;
    k_x -= max_wave_number_;
    for (int idx_2 = 0; idx_2 < stability_matrix_rank_; ++idx_2)
    {
      OneDimIdxToThreeDimIdx(idx_2, m, q_x, q_y);
      m -= max_mode_idx_;
      q_x -= max_wave_number_;
      matrix_element.real(0.0), matrix_element.imag(0.0);

      bool is_nonzero = false;
      if (n == m)
      {
        if ((k_x == q_x) && (k_y != q_y))
        {
          matrix_element += Real(k_x) / Real(k_y - q_y);
          if (q_y != 0)
          {
            matrix_element += Real(k_x) / Real(k_y - q_y);
          }
          is_nonzero = true;
        }
        if ((k_x != q_x) && (k_y == q_y))
        {
          matrix_element -= Real(k_y) / Real(k_x - q_x);
          is_nonzero = true;
        }
      }

      // raw major order
//      raw_stability_matrix_[idx_2 + idx_1 * stability_matrix_rank_] = matrix_element;
      if (is_nonzero)
      {
        armadillo_stability_matrix_(idx_1, idx_2) = matrix_element;
      }
    } // idx_2
  } // idx_1
}