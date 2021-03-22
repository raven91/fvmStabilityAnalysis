//
// Created by Nikita Kruk on 2019-06-24.
//

#include "NonhomogeneousPerturbationNonzeroLag.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator> // std::ostream_iterator, std::copy
#include <algorithm> // std::copy, std::fill, std::count_if
#include <cassert>
#include <chrono>
#include <thread>
#include <mutex>

#include <boost/math/special_functions/bessel.hpp>

#include <mkl.h>
#include <eigen3/Eigen/Eigenvalues>
//#include </home/nkruk/libs/eigen3/include/eigen3/Eigen/Eigenvalues>

std::mutex output_mutex;
const Real scaler = 1.0;

NonhomogeneousPerturbationNonzeroLag::NonhomogeneousPerturbationNonzeroLag(Thread *thread,
                                                                           bool save_only_max_eigenvalues) :
    thread_(thread),
    max_mode_idx_(100),
    max_wave_number_(10), // 30 used for PRL paper
    eigen_stability_matrices_(kWaveNumberThreads, Eigen::MatrixXcd::Zero(2 * max_mode_idx_ + 1, 2 * max_mode_idx_ + 1)),
    raw_stability_matrices_(kWaveNumberThreads,
                            std::vector<Complex>((2 * max_mode_idx_ + 1) * (2 * max_mode_idx_ + 1), Complex(0.0, 0.0))),
    eigenvalues_per_thread_(kWaveNumberThreads, std::vector<Complex>(2 * max_mode_idx_ + 1, Complex(0.0, 0.0))),
    fourier_modes_sp_(2 * max_mode_idx_ + 1, Complex(0.0, 0.0)),
    fourier_modes_mp_(2 * max_mode_idx_ + 1, MultiprecisionComplex(0.0, 0.0)),
    max_bessel_idx_(2000), // 1000 used for PRL paper
    bessel_functions_mp_(max_mode_idx_ + max_bessel_idx_ + 1, MultiprecisionReal(0.0)),
    save_only_max_eigenvalues_(save_only_max_eigenvalues),
    bessel_function_table_(),
    counter_for_bessel_functions_(0),
    counter_for_wave_numbers_(0)
{

}

NonhomogeneousPerturbationNonzeroLag::~NonhomogeneousPerturbationNonzeroLag() = default;

void NonhomogeneousPerturbationNonzeroLag::StabilityOfUniformSolution()
{
  std::ofstream stability_analysis_file;
  CreateOutputFileForUniformSolution(stability_analysis_file);

  const int n_wavenumber = 2 * max_wave_number_ * max_wave_number_;
  std::vector<Complex> max_eigenvalue_for_fixed_wavenumber(n_wavenumber, Complex(0.0, 0.0));
  Complex max_max_eigenvalue(0.0, 0.0);

  const Real scale = 1.0;
  Real sigma(1.0), D_phi(0.0025), alpha(1.54);
  const std::vector<Real> rhos{0.06};//{-1.0, 0.1/scale, 0.25/scale, 0.5/scale};
  const std::vector<Real> v_0s{0.1};
  /*if (!save_only_max_eigenvalues_)
  {
    std::cout << "rho:>";
    std::cin >> rho;
    rhos = std::vector<Real>(1, rho);
  }*/
#if defined(BCS_CLUSTER)
  const Real D_phi_min = 0.0;
  const Real D_phi_max = 1.0;
  const Real dD_phi = 0.025;
  const int n_D_phi = std::nearbyint((D_phi_max - D_phi_min) / dD_phi);
  const Real dalpha = 0.01;
  const int n_alpha = 16;
  const Real alpha_min = 0.0 + kRank * n_alpha * dalpha;
  const Real alpha_max = 0.0 + (kRank + 1) * n_alpha * dalpha;
#else
  const Real D_phi_min = 0.0;
  const Real D_phi_max = 0.5;
  const Real dD_phi = 0.025;
  const int n_D_phi = std::nearbyint((D_phi_max - D_phi_min) / dD_phi);
  const Real alpha_min = 0.0;
  const Real alpha_max = 1.6;
  const Real dalpha = 0.01;
  const int n_alpha = std::nearbyint((alpha_max - alpha_min) / dalpha);
#endif
  std::chrono::time_point<std::chrono::system_clock> timer = std::chrono::system_clock::now();

//  for (int i_D_phi = 0; i_D_phi <= n_D_phi; ++i_D_phi)
  {
//    D_phi = D_phi_min + i_D_phi * dD_phi;
//    for (int i_alpha = 0; i_alpha <= n_alpha; ++i_alpha)
    {
//      alpha = alpha_min + i_alpha * dalpha;
      /*if (D_phi < 0.5 * sigma * std::cos(alpha)) // the region of incoherence
      {
        continue;
      }*/
      for (const Real v_0 : v_0s)
      {
        for (const Real rho : rhos)
        {
          std::ostringstream output_string;
          output_string << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi << '\t';
          std::fill(max_eigenvalue_for_fixed_wavenumber.begin(),
                    max_eigenvalue_for_fixed_wavenumber.end(),
                    Complex(0.0, 0.0));

          counter_for_wave_numbers_ = 0;
          std::vector<std::thread> wave_number_threads(kWaveNumberThreads);
          for (int thread_id = 0; thread_id < kWaveNumberThreads; ++thread_id)
          {
            wave_number_threads[thread_id] =
                std::thread(&NonhomogeneousPerturbationNonzeroLag::FindMaxEigenvaluesForUniformDensity,
                            this,
                            thread_id,
                            v_0,
                            sigma,
                            rho,
                            alpha,
                            D_phi,
                            n_wavenumber,
                            std::ref(max_eigenvalue_for_fixed_wavenumber),
                            std::ref(stability_analysis_file));
          } // thread_id
          for (auto &wave_number_thread : wave_number_threads)
          {
            wave_number_thread.join();
          } // wave_number_thread

          if (save_only_max_eigenvalues_)
          {
            for (int k = 0; k < n_wavenumber; ++k)
            {
              output_string << max_eigenvalue_for_fixed_wavenumber[k].real() << '\t'
                            << max_eigenvalue_for_fixed_wavenumber[k].imag() << '\t';
            } // k
            stability_analysis_file << output_string.str() << std::endl;

            max_max_eigenvalue = *std::max_element(max_eigenvalue_for_fixed_wavenumber.begin(),
                                                   max_eigenvalue_for_fixed_wavenumber.end(),
                                                   CompareComplexByRealPart());
            std::cout << "[" << kRank << "]\t" << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi
                      << '\t' << max_max_eigenvalue << std::endl;
          }
        } // rho
      } // v_0
      std::chrono::duration<Real> elapsed_seconds = std::chrono::system_clock::now() - timer;
      std::cout << "time per cycle: " << elapsed_seconds.count() << "s" << std::endl;
      timer = std::chrono::system_clock::now();
    } // i_alpha
  } // i_D_phi

  stability_analysis_file.close();
}

void NonhomogeneousPerturbationNonzeroLag::FindMaxEigenvaluesForUniformDensity(int thread_id,
                                                                               Real v_0,
                                                                               Real sigma,
                                                                               Real rho,
                                                                               Real alpha,
                                                                               Real D_phi,
                                                                               int n_wavenumber,
                                                                               std::vector<Complex> &max_eigenvalue_for_fixed_wavenumber,
                                                                               std::ofstream &stability_analysis_file)
{
//  const Real wavenumber_min = 0.0, wavenumber_max = 100.0;
//  const Real d_wavenumber = (wavenumber_max - wavenumber_min) / n_wavenumber;
//  const int n_max_wavenumber_to_save = 1;
  Real j_1 = 0.0;
  Real k_x = 0.0, k_y = 0.0, k_abs = 0.0;
  Complex max_eigenvalue(0.0, 0.0);
  Real j_arg(0.0);

  int k = std::atomic_fetch_add(&counter_for_wave_numbers_, int(1));
  while (k < n_wavenumber)
//  for (int k = 0; k <= n_wavenumber; ++k)
  {
//      k_x = wavenumber_min + k * d_wavenumber;
//      k_y = wavenumber_min + k * d_wavenumber;
//      k_abs = std::sqrt(k_x * k_x + k_y * k_y);
    k_x = (k % (2 * max_wave_number_)) - max_wave_number_;
    k_y = k / (2 * max_wave_number_);
    const Real eps_dk = 0.005;
    if ((std::fabs(k_x) < eps_dk && std::fabs(k_y) < eps_dk) || (rho < 0.0)) // k\lim_\rightarrow0, \rho\lim\rightarrow0
    { // negative value of rho is indicative of the hydrodynamic limit
      j_1 = 1.0;
    } else
    {
      k_abs = std::hypot(k_x, k_y);
      j_arg = kTwoPi * rho * k_abs;
      j_1 = Real(2.0 * boost::math::cyl_bessel_j(1, j_arg) / j_arg);
    }
    FillInStabilityMatrixForUniformSolution(thread_id, k_x, k_y, v_0, sigma, alpha, D_phi, j_1);

    /*eigen_stability_matrix_ =
        Eigen::Map<Eigen::MatrixXcd>(raw_stability_matrix_.data(), 2 * max_mode_idx_ + 1, 2 * max_mode_idx_ + 1);
    eigen_stability_matrix_.transposeInPlace();
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigen_solver(eigen_stability_matrix_, false);
    const Eigen::VectorXcd &eigenvalues = eigen_solver.eigenvalues();
    Eigen::VectorXcd::Map(&eigenvalues_[0], eigenvalues_.size()) = eigenvalues;*/
    int sdim = 0;
    LAPACKE_zgees(LAPACK_ROW_MAJOR,
                  'N',
                  'N',
                  nullptr,
                  2 * max_mode_idx_ + 1,
                  &raw_stability_matrices_[thread_id][0],
                  2 * max_mode_idx_ + 1,
                  &sdim,
                  &eigenvalues_per_thread_[thread_id][0],
                  nullptr,
                  2 * max_mode_idx_ + 1);
    std::for_each(eigenvalues_per_thread_[thread_id].begin(),
                  eigenvalues_per_thread_[thread_id].end(),
                  [&](Complex &c) { c /= scaler; });
    std::sort(eigenvalues_per_thread_[thread_id].begin(),
              eigenvalues_per_thread_[thread_id].end(),
              CompareComplexByRealPart());
    max_eigenvalue = eigenvalues_per_thread_[thread_id].back();
    max_eigenvalue_for_fixed_wavenumber[k] = max_eigenvalue;
    if (!save_only_max_eigenvalues_)
    {
      std::cout << "v_0:" << v_0 << ", sigma:" << sigma << ", rho:" << rho << ", D_phi:" << D_phi << ", alpha:" << alpha
                << ", wavenumber:(" << k_x << "," << k_y << "), max_lambda:" << max_eigenvalue << std::endl;
      std::ostringstream local_output_string;
      local_output_string << k_x << '\t' << k_y << '\t';
      for (const Complex &eigenvalue : eigenvalues_per_thread_[thread_id])
      {
        local_output_string << eigenvalue.real() << '\t' << eigenvalue.imag() << '\t';
      } // eigenvalue
      output_mutex.lock();
      stability_analysis_file << local_output_string.str() << std::endl;
      output_mutex.unlock();
    } else
    {
      /*for (int i = eigenvalues_.size() - n_max_wavenumber_to_save; i < eigenvalues_.size(); ++i)
      {
        output_string << eigenvalues_[i].real() << '\t' << eigenvalues_[i].imag() << '\t';
      } // i*/
//      output_string << max_eigenvalue.real() << '\t' << max_eigenvalue.imag() << '\t';
    }
    k = std::atomic_fetch_add(&counter_for_wave_numbers_, int(1));
  } // k
}

void NonhomogeneousPerturbationNonzeroLag::CreateOutputFileForUniformSolution(std::ofstream &stability_analysis_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string folder("/home/nkruk/cpp/fvmStabilityAnalysis/output/UniformSolution/");
#else
  std::string folder("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/UniformSolution/");
#endif

#if defined(BCS_CLUSTER)
  stability_analysis_file.open(
      folder + std::string("nonhomogeneous_perturbation_nonzero_lag_s") + std::to_string(kRank) + std::string(".txt"),
      std::ios::out | std::ios::trunc);
#else
  stability_analysis_file.open(folder + std::string("nonhomogeneous_perturbation_nonzero_lag.txt"),
                               std::ios::out | std::ios::trunc);
#endif
  assert(stability_analysis_file.is_open());
}

void NonhomogeneousPerturbationNonzeroLag::FillInStabilityMatrixForUniformSolution(int thread_id,
                                                                                   Real k_x,
                                                                                   Real k_y,
                                                                                   Real v_0,
                                                                                   Real sigma,
                                                                                   Real alpha,
                                                                                   Real D_phi,
                                                                                   Real j_1)
{
  const Complex exp_plus_i_alpha = std::exp(kI * alpha), exp_minus_i_alpha = std::exp(-kI * alpha);

  std::vector<Complex> &raw_stability_matrix = raw_stability_matrices_[thread_id];
  Complex matrix_element(0.0, 0.0);
  int nn = 0, mm = 0;
  for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
  {
    nn = max_mode_idx_ + n;
    for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
    {
      mm = max_mode_idx_ + m;
      matrix_element.real(0.0), matrix_element.imag(0.0);
      if (n - 1 == m)
      {
        matrix_element += 0.5 * v_0 * (kI * k_x - k_y);
      }
      if (n + 1 == m)
      {
        matrix_element += 0.5 * v_0 * (kI * k_x + k_y);
      }
      if (n == m)
      {
        if (n == 1)
        {
          matrix_element += 0.5 * n * sigma * j_1 * exp_minus_i_alpha;
        }
        if (n == -1)
        {
          matrix_element -= 0.5 * n * sigma * j_1 * exp_plus_i_alpha;
        }
        matrix_element -= n * n * D_phi;
      }

//      eigen_stability_matrix_(nn, mm) = matrix_element;
      raw_stability_matrix[mm + nn * (2 * max_mode_idx_ + 1)] = matrix_element;
    } // m
  } // n
  std::for_each(raw_stability_matrix.begin(), raw_stability_matrix.end(), [&](Complex &c) { c *= scaler; });
}

void NonhomogeneousPerturbationNonzeroLag::FillInStabilityMatrixForUniformSolutionAgainstZeroWavenumber(Real sigma,
                                                                                                        Real alpha,
                                                                                                        Real D_phi)
{
  /*const Complex exp_plus_i_alpha = std::exp(kI * alpha), exp_minus_i_alpha = std::exp(-kI * alpha);

  Complex matrix_element(0.0, 0.0);
  int nn = 0, mm = 0;
  for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
  {
    nn = max_mode_idx_ + n;
    for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
    {
      mm = max_mode_idx_ + m;
      matrix_element.real(0.0), matrix_element.imag(0.0);
      if (n == m)
      {
        if (n == 1)
        {
          matrix_element += 0.5 * n * sigma * exp_minus_i_alpha;
        }
        if (n == -1)
        {
          matrix_element -= 0.5 * n * sigma * exp_plus_i_alpha;
        }
        matrix_element -= D_phi * n * n;
      }

//      eigen_stability_matrix_(nn, mm) = matrix_element;
      raw_stability_matrices_[0][mm + nn * (2 * max_mode_idx_ + 1)] = matrix_element;
    } // m
  } // n*/
}

void NonhomogeneousPerturbationNonzeroLag::StabilityOfNontrivialSolution()
{
  std::ifstream order_parameter_file, velocity_file;
  CreateInputFilesForNontrivialSolution(order_parameter_file, velocity_file);
  std::ofstream stability_analysis_file, second_stability_analysis_file, unstable_modes_file;
  CreateOutputFileForNontrivialSolution(stability_analysis_file, second_stability_analysis_file, unstable_modes_file);
//  InitializeBesselFunctionTable();

  const int n_wavenumber = 2 * max_wave_number_ * max_wave_number_;
  std::vector<int> number_of_unstable_modes_for_fixed_wavenumber(n_wavenumber, 0);
  std::vector<Complex> max_eigenvalue_for_fixed_wavenumber(n_wavenumber, Complex(0.0, 0.0));
  std::vector<Complex> second_max_eigenvalue_for_fixed_wavenumber(n_wavenumber, Complex(0.0, 0.0));
  Complex max_max_eigenvalue(0.0, 0.0);

  const Real scale = 1.0;
  Real sigma(0.0), D_phi(0.0), alpha(0.0), order_parameter(0.0), velocity(0.0);
  const std::vector<Real> rhos{0.3};//{0.1/scale, 1.0/scale, 10.0/scale};
  const std::vector<Real> v_0s{1.0/1.0};//{0.001, 0.01, 0.1, 1.0};
  /*if (!save_only_max_eigenvalues_)
  {
    std::cout << "rho:>";
    std::cin >> rho;
    rhos = std::vector<Real>(1, rho);
  }*/
  /*for (int i = 1; i <= 200; ++i)
  {
    rhos.push_back(0.25 * i / scale);
  }*/
  static MultiprecisionReal gamma = 0.0;

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
    const Real eps_param = 0.001;
    if (//(D_phi < 0.005 - eps_param) || (D_phi > 0.005 + eps_param && D_phi < 0.01 - eps_param) || (D_phi > 0.01 + eps_param) ||
        (D_phi > 0.5 * sigma * std::cos(alpha)) // the region of incoherence is not valid for the Bessel decomposition
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
        bessel_threads[i] = std::thread(&NonhomogeneousPerturbationNonzeroLag::CalculateModifiedBesselFunctions,
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
        std::ostringstream output_string, second_output_string, unstable_modes_string;
        output_string << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi << '\t';
        second_output_string << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi << '\t';
        unstable_modes_string << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi << '\t';
        std::fill(max_eigenvalue_for_fixed_wavenumber.begin(),
                  max_eigenvalue_for_fixed_wavenumber.end(),
                  Complex(0.0, 0.0));
        std::fill(second_max_eigenvalue_for_fixed_wavenumber.begin(),
                  second_max_eigenvalue_for_fixed_wavenumber.end(),
                  Complex(0.0, 0.0));
        std::fill(number_of_unstable_modes_for_fixed_wavenumber.begin(),
                  number_of_unstable_modes_for_fixed_wavenumber.end(),
                  0);

        counter_for_wave_numbers_ = 0;
        std::vector<std::thread> wave_number_threads(kWaveNumberThreads);
        for (int thread_id = 0; thread_id < kWaveNumberThreads; ++thread_id)
        {
          wave_number_threads[thread_id] =
              std::thread(&NonhomogeneousPerturbationNonzeroLag::FindMaxEigenvaluesForNontrivialDensity,
                          this,
                          thread_id,
                          v_0,
                          sigma,
                          rho,
                          alpha,
                          D_phi,
                          order_parameter,
                          velocity,
                          n_wavenumber,
                          std::ref(number_of_unstable_modes_for_fixed_wavenumber),
                          std::ref(max_eigenvalue_for_fixed_wavenumber),
                          std::ref(second_max_eigenvalue_for_fixed_wavenumber),
                          std::ref(stability_analysis_file));
        } // thread_id
        for (auto &wave_number_thread : wave_number_threads)
        {
          wave_number_thread.join();
        } // wave_number_thread

        if (save_only_max_eigenvalues_)
        {
          for (int k = 0; k < n_wavenumber; ++k)
          {
            output_string << max_eigenvalue_for_fixed_wavenumber[k].real() << '\t'
                          << max_eigenvalue_for_fixed_wavenumber[k].imag() << '\t';
            second_output_string << second_max_eigenvalue_for_fixed_wavenumber[k].real() << '\t'
                                 << second_max_eigenvalue_for_fixed_wavenumber[k].imag() << '\t';
            unstable_modes_string << number_of_unstable_modes_for_fixed_wavenumber[k] << '\t';
          } // k
          stability_analysis_file << output_string.str() << std::endl;
          second_stability_analysis_file << second_output_string.str() << std::endl;
          unstable_modes_file << unstable_modes_string.str() << std::endl;

          max_max_eigenvalue = *std::max_element(max_eigenvalue_for_fixed_wavenumber.begin(),
                                                 max_eigenvalue_for_fixed_wavenumber.end(),
                                                 CompareComplexByRealPart());
          std::cout << "[" << kRank << "]\t" << v_0 << '\t' << sigma << '\t' << rho << '\t' << alpha << '\t' << D_phi
                    << '\t' << max_max_eigenvalue << std::endl;
        }
      } // rho
    } // v_0
    std::chrono::duration<Real> elapsed_seconds = std::chrono::system_clock::now() - timer;
    std::cout << "time per cycle: " << elapsed_seconds.count() << "s" << std::endl;
    timer = std::chrono::system_clock::now();
  } // read new parameters

  stability_analysis_file.close();
  second_stability_analysis_file.close();
  unstable_modes_file.close();
  velocity_file.close();
  order_parameter_file.close();
}

void NonhomogeneousPerturbationNonzeroLag::FindMaxEigenvaluesForNontrivialDensity(int thread_id,
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
                                                                                  std::ofstream &stability_analysis_file)
{
  //  const Real wavenumber_min = 0.0, wavenumber_max = 10.0;
  //  const Real d_wavenumber = (wavenumber_max - wavenumber_min) / n_wavenumber;
  Real j_1 = 0.0;
  Real k_x = 0.0, k_y = 0.0, k_abs = 0.0;
  Complex max_eigenvalue(0.0, 0.0), second_max_eigenvalue(0.0, 0.0);
  Real j_arg(0.0);

  int k = std::atomic_fetch_add(&counter_for_wave_numbers_, int(1));
  while (k < n_wavenumber)
//  for (int k = 0; k < n_wavenumber; ++k)
  {
    k_x = (k % (2 * max_wave_number_)) - max_wave_number_;
    k_y = k / (2 * max_wave_number_);
    const Real eps_dk = 0.005;
    if ((std::fabs(k_x) < eps_dk && std::fabs(k_y) < eps_dk) || (rho < 0.0)) // k\lim_\rightarrow0, \rho\lim\rightarrow0
    { // negative value of rho is indicative of the hydrodynamic limit
      j_1 = 1.0;
    } else
    {
      k_abs = std::sqrt(k_x * k_x + k_y * k_y);
      j_arg = kTwoPi * rho * k_abs;
      j_1 = Real(2.0 * boost::math::cyl_bessel_j(1, j_arg) / j_arg);
    }
    FillInStabilityMatrixForNontrivialSolution(thread_id, k_x, k_y, v_0, sigma, alpha, D_phi, velocity, j_1);

    /*if (std::any_of(raw_stability_matrices_[thread_id].begin(),
                    raw_stability_matrices_[thread_id].end(),
                    [](const Complex &c) { return (std::isnan(c.real()) || std::isnan(c.imag())); }))
    {
      std::fill(eigenvalues_per_thread_[thread_id].begin(),
                eigenvalues_per_thread_[thread_id].end(),
                Complex(std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::quiet_NaN()));
    } else*/
    {
      /*eigen_stability_matrices_[thread_id] =
          Eigen::Map<Eigen::MatrixXcd>(raw_stability_matrices_[thread_id].data(), 2 * max_mode_idx_ + 1, 2 * max_mode_idx_ + 1);
      eigen_stability_matrices_[thread_id].transposeInPlace();
      Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigen_solver(eigen_stability_matrices_[thread_id], false);
      const Eigen::VectorXcd &eigenvalues = eigen_solver.eigenvalues();
      Eigen::VectorXcd::Map(&eigenvalues_per_thread_[thread_id][0], eigenvalues_per_thread_[thread_id].size()) = eigenvalues;*/
      int sdim = 0;
      LAPACKE_zgees(LAPACK_ROW_MAJOR,
                    'N',
                    'N',
                    nullptr,
                    2 * max_mode_idx_ + 1,
                    &raw_stability_matrices_[thread_id][0],
                    2 * max_mode_idx_ + 1,
                    &sdim,
                    &eigenvalues_per_thread_[thread_id][0],
                    nullptr,
                    2 * max_mode_idx_ + 1);
      std::for_each(eigenvalues_per_thread_[thread_id].begin(),
                    eigenvalues_per_thread_[thread_id].end(),
                    [&](Complex &c) { c /= scaler; });
    }
    number_of_unstable_modes_for_fixed_wavenumber[k] = std::count_if(eigenvalues_per_thread_[thread_id].begin(),
                                                                     eigenvalues_per_thread_[thread_id].end(),
                                                                     [](const Complex &c) { return (c.real() > 0.0); });
    std::sort(eigenvalues_per_thread_[thread_id].begin(),
              eigenvalues_per_thread_[thread_id].end(),
              CompareComplexByRealPart());
    max_eigenvalue = eigenvalues_per_thread_[thread_id][eigenvalues_per_thread_[thread_id].size() - 1];
    second_max_eigenvalue = eigenvalues_per_thread_[thread_id][eigenvalues_per_thread_[thread_id].size() - 2];
    max_eigenvalue_for_fixed_wavenumber[k] = max_eigenvalue;
    second_max_eigenvalue_for_fixed_wavenumber[k] = second_max_eigenvalue;
    if (!save_only_max_eigenvalues_)
    {
      std::cout << "v_0:" << v_0 << ", sigma:" << sigma << ", rho:" << rho << ", D_phi:" << D_phi << ", alpha:" << alpha
                << ", R:" << order_parameter << ", v:" << velocity
                << ", wavenumber:(" << k_x << "," << k_y << "), max_lambda:" << max_eigenvalue << std::endl;
      std::ostringstream local_output_string;
      local_output_string << k_x << '\t' << k_y << '\t';
      for (const Complex &eigenvalue : eigenvalues_per_thread_[thread_id])
      {
        local_output_string << eigenvalue.real() << '\t' << eigenvalue.imag() << '\t';
      } // eigenvalue
      output_mutex.lock();
      stability_analysis_file << local_output_string.str() << std::endl;
      output_mutex.unlock();
    } /*else
    {
      for (int i = eigenvalues_per_thread_[thread_id].size() - n_max_wavenumber_to_save; i < eigenvalues_per_thread_[thread_id].size(); ++i)
      {
        output_string << eigenvalues_per_thread_[thread_id][i].real() << '\t' << eigenvalues_per_thread_[thread_id][i].imag() << '\t';
      } // i
    }*/
    k = std::atomic_fetch_add(&counter_for_wave_numbers_, int(1));
  } // k
}

void NonhomogeneousPerturbationNonzeroLag::CreateInputFilesForNontrivialSolution(std::ifstream &order_parameter_file,
                                                                                 std::ifstream &velocity_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string input_folder("/home/nkruk/cpp/fvmSelfconsistentEquations/output/HomogeneousSolutionsNonzeroLag/");
#elif defined(__APPLE__)
  std::string input_folder
      ("/Users/nikita/Documents/Projects/fvm/fvmSelfconsistentEquations/HomogeneousSolutionsNonzeroLag/");
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

void NonhomogeneousPerturbationNonzeroLag::CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file,
                                                                                 std::ofstream &second_stability_analysis_file,
                                                                                 std::ofstream &unstable_modes_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string output_folder("/home/nkruk/cpp/fvmStabilityAnalysis/output/NontrivialSolution/periodic/scale100/");
#elif defined(__APPLE__)
  std::string output_folder("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/NontrivialSolution/");
#endif

#if defined(BCS_CLUSTER)
  stability_analysis_file.open(
      output_folder + std::string("nonhomogeneous_perturbation_nonzero_lag_nd") + std::to_string(kRank)
          + std::string(".txt"), std::ios::out | std::ios::trunc);
  second_stability_analysis_file.open(
      output_folder + std::string("second_nonhomogeneous_perturbation_nonzero_lag_nd") + std::to_string(kRank)
          + std::string(".txt"), std::ios::out | std::ios::trunc);
  unstable_modes_file.open(
      output_folder + std::string("number_of_unstable_modes_for_fixed_wave_vector_nd") + std::to_string(kRank)
          + std::string(".txt"), std::ios::out | std::ios::trunc);
#else
  stability_analysis_file.open(output_folder + std::string("nonhomogeneous_perturbation_nonzero_lag.txt"),
                               std::ios::out | std::ios::trunc);
  second_stability_analysis_file.open(output_folder + std::string("second_nonhomogeneous_perturbation_nonzero_lag.txt"),
                                      std::ios::out | std::ios::trunc);
  unstable_modes_file.open(output_folder + std::string("number_of_unstable_modes_for_fixed_wave_vector.txt"),
                           std::ios::out | std::ios::trunc);
#endif
  assert(stability_analysis_file.is_open());
  assert(second_stability_analysis_file.is_open());
  assert(unstable_modes_file.is_open());
}

void NonhomogeneousPerturbationNonzeroLag::C_1(int n,
                                               const MultiprecisionReal &D_phi,
                                               const MultiprecisionReal &velocity,
                                               MultiprecisionComplex &c_1) const
{
  static MultiprecisionReal modulus(0.0);
  modulus = (velocity * velocity) / (D_phi * D_phi) + n * n;
  c_1.real(velocity / D_phi / modulus);
  c_1.imag(n / modulus);
}

void NonhomogeneousPerturbationNonzeroLag::InitializeBesselFunctionTable()
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

bool NonhomogeneousPerturbationNonzeroLag::QueryModifiedBesselFunctionsFromTable(Real sigma, Real D_phi, Real alpha)
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

void NonhomogeneousPerturbationNonzeroLag::CalculateModifiedBesselFunctions(const MultiprecisionReal &gamma)
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

void NonhomogeneousPerturbationNonzeroLag::ComputeFourierModes(Real alpha,
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
  normalization_constant = 1.0 / normalization_constant.real();
//  int i = 0;
  std::for_each(&fourier_modes_mp_[max_mode_idx_ + 0],
                &fourier_modes_mp_[2 * max_mode_idx_ + 1],
                [&](MultiprecisionComplex &f)
                {
                  f *= normalization_constant.real();
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

void NonhomogeneousPerturbationNonzeroLag::FillInStabilityMatrixForNontrivialSolution(int thread_id,
                                                                                      Real k_x,
                                                                                      Real k_y,
                                                                                      Real v_0,
                                                                                      Real sigma,
                                                                                      Real alpha,
                                                                                      Real D_phi,
                                                                                      Real velocity,
                                                                                      Real j_1)
{
  const Complex &f_0 = fourier_modes_sp_[max_mode_idx_ + 0];
  const Complex f_0_inverse = 1.0 / f_0;
  const Complex &f_1 = fourier_modes_sp_[max_mode_idx_ + 1];
  const Complex &f_minus_1 = fourier_modes_sp_[max_mode_idx_ - 1];
  const Complex exp_plus_i_alpha = std::exp(kI * alpha), exp_minus_i_alpha = std::exp(-kI * alpha);

  std::vector<Complex> &raw_stability_matrix = raw_stability_matrices_[thread_id];
  Complex matrix_element(0.0, 0.0);
  int nn = 0, mm = 0;
  for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
  {
    nn = max_mode_idx_ + n;
    for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
    {
      mm = max_mode_idx_ + m;
      matrix_element.real(0.0), matrix_element.imag(0.0);
      if (n - 1 == m)
      {
        matrix_element += 0.5 * v_0 * kTwoPi * (kI * k_x - k_y);
        matrix_element += 0.5 * n * sigma * f_0_inverse * f_1 * exp_minus_i_alpha;
      }
      if (n + 1 == m)
      {
        matrix_element += 0.5 * v_0 * kTwoPi * (kI * k_x + k_y);
        matrix_element -= 0.5 * n * sigma * f_0_inverse * f_minus_1 * exp_plus_i_alpha;
      }
      if (n == m)
      {
        matrix_element += -(n * n * D_phi + kI * Real(n) * velocity) + kI * velocity * kPi * (k_x - k_y);
      }
      if (1 == m)
      {
        matrix_element +=
            0.5 * n * sigma * f_0_inverse * j_1 * fourier_modes_sp_[max_mode_idx_ + n - 1] * exp_minus_i_alpha;
      }
      if (-1 == m)
      {
        matrix_element -=
            0.5 * n * sigma * f_0_inverse * j_1 * fourier_modes_sp_[max_mode_idx_ + n + 1] * exp_plus_i_alpha;
      }
      if (0 == m)
      {
        matrix_element -=
            0.5 * n * sigma * f_0_inverse * j_1 * f_1 * fourier_modes_sp_[max_mode_idx_ + n - 1] * f_0_inverse
                * exp_minus_i_alpha;
        matrix_element +=
            0.5 * n * sigma * f_0_inverse * j_1 * f_minus_1 * fourier_modes_sp_[max_mode_idx_ + n + 1] * f_0_inverse
                * exp_plus_i_alpha;
      }

      raw_stability_matrix[mm + nn * (2 * max_mode_idx_ + 1)] = matrix_element;
    } // m
  } // n
  std::for_each(raw_stability_matrix.begin(), raw_stability_matrix.end(), [&](Complex &c) { c *= scaler; });
}

void NonhomogeneousPerturbationNonzeroLag::FillInStabilityMatrixForNontrivialSolution(int thread_id,
                                                                                      Real k_x,
                                                                                      Real k_y,
                                                                                      Real v_0,
                                                                                      Real sigma,
                                                                                      Real alpha,
                                                                                      Real D_phi,
                                                                                      Real velocity,
                                                                                      const MultiprecisionReal &j_1)
{
//  const MultiprecisionComplex &f_0 = fourier_modes_mp_[max_mode_idx_ + 0];
//  const MultiprecisionComplex f_0_inverse = 1.0 / f_0;
//  const MultiprecisionComplex &f_1 = fourier_modes_mp_[max_mode_idx_ + 1];
//  const MultiprecisionComplex &f_minus_1 = fourier_modes_mp_[max_mode_idx_ - 1];
//  MultiprecisionComplex c_alpha(alpha, 0.0), exp_plus_i_alpha(0.0, 0.0), exp_minus_i_alpha(0.0, 0.0);
//  UnitExp(c_alpha, exp_plus_i_alpha);
//  UnitExp(-c_alpha, exp_minus_i_alpha);
//
//  std::vector<Complex> &raw_stability_matrix = raw_stability_matrices_[thread_id];
//  Complex matrix_element(0.0, 0.0);
//  int nn = 0, mm = 0;
//  for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
//  {
//    nn = max_mode_idx_ + n;
//    for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
//    {
//      mm = max_mode_idx_ + m;
//      matrix_element.real(0.0), matrix_element.imag(0.0);
//      if (n - 1 == m)
//      {
//        matrix_element += 0.5 * v_0 * kTwoPi * (kI * k_x - k_y);
//        matrix_element += 0.5 * n * sigma * f_0_inverse * f_1 * exp_minus_i_alpha;
//      }
//      if (n + 1 == m)
//      {
//        matrix_element += 0.5 * v_0 * kTwoPi * (kI * k_x + k_y);
//        matrix_element -= 0.5 * n * sigma * f_0_inverse * f_minus_1 * exp_plus_i_alpha;
//      }
//      if (n == m)
//      {
//        matrix_element += -(n * n * D_phi + kI * Real(n) * velocity) + kI * velocity * kPi * (k_x - k_y);
//      }
//      if (1 == m)
//      {
//        matrix_element +=
//            0.5 * n * sigma * f_0_inverse * j_1 * fourier_modes_sp_[max_mode_idx_ + n - 1] * exp_minus_i_alpha;
//      }
//      if (-1 == m)
//      {
//        matrix_element -=
//            0.5 * n * sigma * f_0_inverse * j_1 * fourier_modes_sp_[max_mode_idx_ + n + 1] * exp_plus_i_alpha;
//      }
//      if (0 == m)
//      {
//        matrix_element -=
//            0.5 * n * sigma * f_0_inverse * j_1 * f_1 * fourier_modes_sp_[max_mode_idx_ + n - 1] * f_0_inverse
//                * exp_minus_i_alpha;
//        matrix_element +=
//            0.5 * n * sigma * f_0_inverse * j_1 * f_minus_1 * fourier_modes_sp_[max_mode_idx_ + n + 1] * f_0_inverse
//                * exp_plus_i_alpha;
//      }
//
//      raw_stability_matrix[mm + nn * (2 * max_mode_idx_ + 1)] = matrix_element;
//    } // m
//  } // n
}