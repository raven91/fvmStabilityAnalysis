//
// Created by Nikita Kruk on 2019-06-24.
//

#include "NonhomogeneousPerturbationZeroLag.hpp"

#include <iostream>
#include <fstream>
#include <iterator> // std::ostream_iterator
#include <algorithm> // std::copy
#include <cassert>
#include <chrono>
#include <thread>

#include <boost/math/special_functions/bessel.hpp>

#include <eigen3/Eigen/Eigenvalues>

NonhomogeneousPerturbationZeroLag::NonhomogeneousPerturbationZeroLag(bool save_only_max_eigenvalues) :
    max_mode_idx_(100),
    max_wave_number_(30),
    eigen_stability_matrices_(kWaveNumberThreads, Eigen::MatrixXcd::Zero(2 * max_mode_idx_ + 1, 2 * max_mode_idx_ + 1)),
    eigenvalues_per_thread_(kWaveNumberThreads, std::vector<Complex>(2 * max_mode_idx_ + 1, Complex(0.0, 0.0))),
    save_only_max_eigenvalues_(save_only_max_eigenvalues)
{

}

NonhomogeneousPerturbationZeroLag::~NonhomogeneousPerturbationZeroLag()
{

}

void NonhomogeneousPerturbationZeroLag::StabilityOfNontrivialSolution()
{
  std::ifstream order_parameter_file;
  CreateInputFileForNontrivialSolution(order_parameter_file);
  std::ofstream stability_analysis_file;
  CreateOutputFileForNontrivialSolution(stability_analysis_file);

  const int n_wavenumber = 2 * max_wave_number_ * max_wave_number_;
  const Real wavenumber_min = 0.0, wavenumber_max = 100.0;
  const Real d_wavenumber = (wavenumber_max - wavenumber_min) / n_wavenumber;
  std::vector<Complex> max_eigenvalue_for_fixed_wavenumber(n_wavenumber + 1, Complex(0.0, 0.0));
  std::vector<Complex> second_max_eigenvalue_for_fixed_wavenumber(n_wavenumber + 1, Complex(0.0, 0.0));
  Complex max_max_eigenvalue(0.0, 0.0);

  Real coupling_strength = 1.0;
  Real noise = 0.0;
  Real order_parameter = 0.0;
  std::vector<Real> radii_of_interaction{-1.0};//{-1.0, 0.01, 0.1, 0.25, 0.5};
  std::vector<Real> particle_velocities{0.01};//{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0};
  /*if (!save_only_max_eigenvalues_)
  {
    std::cout << "rho:>";
    std::cin >> radius_of_interaction;
    radii_of_interaction = std::vector<Real>(1, radius_of_interaction);
  }*/
  /*for (int i = 1; i <= 250; ++i)
  {
    rhos.push_back(0.2 * i / scale);
  }*/
  MultiprecisionReal gamma = 0.0;
  const MultiprecisionReal two_pi = boost::math::constants::two_pi<MultiprecisionReal>();

  std::chrono::time_point<std::chrono::system_clock> timer = std::chrono::system_clock::now();

//  noise = 0.48;
//  order_parameter = 0.27905934129136;
  while (order_parameter_file >> coupling_strength >> noise >> order_parameter)
  {
    Real eps_param = 0.001;
    if ((noise > 0.5 * coupling_strength) // the region of incoherence is not valid for the Bessel decomposition
        || !std::isfinite(order_parameter))
    {
      continue;
    }
    gamma = coupling_strength * order_parameter / noise;
    std::cout << "gamma:" << gamma << std::endl;

    std::vector<MultiprecisionReal> bessel_functions(max_mode_idx_ + 2, 0.0);
    for (int nu = 0; nu <= max_mode_idx_ + 1; ++nu)
    {
      bessel_functions[nu] = boost::math::cyl_bessel_i(nu, gamma);
//      std::cout << "bessel_i[" << nu << "] " << bessel_functions[nu] << ", f[" << nu << "] "
//                << bessel_functions[nu] / (bessel_functions[0] * two_pi) << std::endl;
    } // n

    for (Real particle_velocity : particle_velocities)
    {
      for (Real radius_of_interaction : radii_of_interaction)
      {
        std::ostringstream output_string, second_output_string;
        output_string << particle_velocity << '\t' << coupling_strength << '\t' << radius_of_interaction << '\t'
                      << noise << '\t';
        second_output_string << particle_velocity << '\t' << coupling_strength << '\t' << radius_of_interaction << '\t'
                             << noise << '\t';
        std::fill(max_eigenvalue_for_fixed_wavenumber.begin(),
                  max_eigenvalue_for_fixed_wavenumber.end(),
                  Complex(0.0, 0.0));
        std::fill(second_max_eigenvalue_for_fixed_wavenumber.begin(),
                  second_max_eigenvalue_for_fixed_wavenumber.end(),
                  Complex(0.0, 0.0));
//        std::vector<Complex> max_eigenvalue_for_fixed_wavenumber(n_wavenumber + 1, Complex(0.0, 0.0));

        counter_for_wave_numbers_ = 0;
        std::vector<std::thread> wave_numbers(kWaveNumberThreads);
        for (int i = 0; i < kWaveNumberThreads; ++i)
        {
          wave_numbers[i] = std::thread(&NonhomogeneousPerturbationZeroLag::FindMaxEigenvaluesForFixedWaveNumber,
                                        this,
                                        i,
                                        particle_velocity,
                                        coupling_strength,
                                        radius_of_interaction,
                                        noise,
                                        order_parameter,
                                        std::ref(bessel_functions),
                                        n_wavenumber,
                                        std::ref(max_eigenvalue_for_fixed_wavenumber),
                                        std::ref(second_max_eigenvalue_for_fixed_wavenumber),
                                        std::ref(stability_analysis_file));
        } // i
        for (auto &wave_number_thread : wave_numbers)
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
          } // k
          max_max_eigenvalue = *std::max_element(max_eigenvalue_for_fixed_wavenumber.begin(),
                                                 max_eigenvalue_for_fixed_wavenumber.end(),
                                                 CompareComplexByRealPart());
          std::cout << particle_velocity << '\t' << coupling_strength << '\t' << radius_of_interaction << '\t' << noise
                    << '\t' << max_max_eigenvalue << std::endl;
//          std::ostringstream local_output_string;
//          local_output_string << particle_velocity << '\t' << coupling_strength << '\t' << radius_of_interaction << '\t'
//                              << noise << '\t';
//          for (const Complex &eigenvalue : max_eigenvalue_for_fixed_wavenumber)
//          {
//            local_output_string << eigenvalue.real() << '\t' << eigenvalue.imag() << '\t';
//          } // eigenvalue
          stability_analysis_file << output_string.str() << std::endl;
        }
      } // rho
    } // particle_velocity
    std::chrono::duration<Real> elapsed_seconds = std::chrono::system_clock::now() - timer;
    std::cout << "time per cycle: " << elapsed_seconds.count() << "s" << std::endl;
    timer = std::chrono::system_clock::now();
  } // read new parameters

  stability_analysis_file.close();
  order_parameter_file.close();
}

void NonhomogeneousPerturbationZeroLag::CreateInputFileForNontrivialSolution(std::ifstream &order_parameter_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string input_folder("/home/nkruk/cpp/fvmSelfconsistentEquations/output/HomogeneousSolutionsZeroLag/");
#elif defined(__APPLE__)
  std::string
      input_folder("/Users/nikita/Documents/Projects/fvm/fvmSelfconsistentEquations/HomogeneousSolutionsZeroLag/");
#endif

  order_parameter_file.open(input_folder + std::string("order_parameter_magnitude_max_digits10.txt"), std::ios::in);
  assert(order_parameter_file.is_open());
}

void NonhomogeneousPerturbationZeroLag::CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file) const
{
#if defined(__linux__) && defined(BCS_CLUSTER)
  std::string output_folder("/home/nkruk/cpp/fvmStabilityAnalysis/output/NontrivialSolution/");
#elif defined(__APPLE__)
  std::string output_folder("/Users/nikita/Documents/Projects/fvm/fvmStabilityAnalysis/NontrivialSolution/");
#endif

  stability_analysis_file.open(output_folder + std::string("nonhomogeneous_perturbation_zero_lag.txt"),
                               std::ios::out | std::ios::trunc);
  assert(stability_analysis_file.is_open());
}

void NonhomogeneousPerturbationZeroLag::FindMaxEigenvaluesForFixedWaveNumber(int thread_id,
                                                                             Real particle_velocity,
                                                                             Real coupling_strength,
                                                                             Real radius_of_interaction,
                                                                             Real noise,
                                                                             Real order_parameter,
                                                                             const std::vector<MultiprecisionReal> &bessel_functions,
                                                                             int n_wavenumber,
                                                                             std::vector<Complex> &max_eigenvalue_for_fixed_wavenumber,
                                                                             std::vector<Complex> &second_max_eigenvalue_for_fixed_wavenumber,
                                                                             std::ofstream &stability_analysis_file)
{
  Real j_1 = 0.0;
  Real k_x = 0.0, k_y = 0.0, k_abs = 0.0;
  Complex max_eigenvalue(0.0, 0.0), second_max_eigenvalue(0.0, 0.0);

  int k = std::atomic_fetch_add(&counter_for_wave_numbers_, int(1));
  while (k < n_wavenumber)
//    for (int k = 0; k <= n_wavenumber; ++k)
  {
    k_x = (k % (2 * max_wave_number_)) - max_wave_number_;
    k_y = k / (2 * max_wave_number_);
    const Real eps_dk = 0.005;
    if ((std::fabs(k_x) < eps_dk && std::fabs(k_y) < eps_dk)
        || (radius_of_interaction < 0.0)) // k\lim_\rightarrow0, \rho\lim\rightarrow0
    {
      j_1 = 1.0;
    } else
    {
//      k_y = wavenumber_min + k * d_wavenumber;
      k_abs = std::sqrt(k_x * k_x + k_y * k_y);
//      j_1 = 2.0 * boost::math::cyl_bessel_j(1, 2.0 * kPi * radius_of_interaction * k_abs) / (2.0 * kPi * radius_of_interaction * k_abs);
      j_1 = boost::math::cyl_bessel_j(0, 2.0 * kPi * radius_of_interaction * k_abs)
          + boost::math::cyl_bessel_j(2, 2.0 * kPi * radius_of_interaction * k_abs);
    }
    FillInStabilityMatrixForNontrivialSolution(thread_id,
                                               k_x,
                                               k_y,
                                               particle_velocity,
                                               coupling_strength,
                                               noise,
                                               j_1,
                                               bessel_functions);

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigen_solver(eigen_stability_matrices_[thread_id], false);
    const Eigen::VectorXcd &eigenvalues = eigen_solver.eigenvalues();
    Eigen::VectorXcd::Map(&eigenvalues_per_thread_[thread_id][0], eigenvalues_per_thread_[thread_id].size()) =
        eigenvalues;
    std::sort(eigenvalues_per_thread_[thread_id].begin(),
              eigenvalues_per_thread_[thread_id].end(),
              CompareComplexByRealPart());
    max_eigenvalue = eigenvalues_per_thread_[thread_id][eigenvalues_per_thread_[thread_id].size() - 1];
    second_max_eigenvalue = eigenvalues_per_thread_[thread_id][eigenvalues_per_thread_[thread_id].size() - 2];
    max_eigenvalue_for_fixed_wavenumber[k] = max_eigenvalue;
    second_max_eigenvalue_for_fixed_wavenumber[k] = second_max_eigenvalue;
    if (!save_only_max_eigenvalues_)
    {
      std::cout << "v_0:" << particle_velocity << ", sigma:" << coupling_strength
                << ", rho:" << radius_of_interaction << ", D_phi:" << noise
                << ", wavenumber:(" << k_x << "," << k_y << "), max_lambda:" << max_eigenvalue << std::endl;
      std::ostringstream local_output_string;
      local_output_string << k_x << '\t' << k_y << '\t';
      for (const Complex &eigenvalue : eigenvalues_per_thread_[thread_id])
      {
        local_output_string << eigenvalue.real() << '\t' << eigenvalue.imag() << '\t';
      } // eigenvalue
      stability_analysis_file << local_output_string.str() << std::endl;
    }
    k = std::atomic_fetch_add(&counter_for_wave_numbers_, int(1));
  } // k
}

void NonhomogeneousPerturbationZeroLag::FillInStabilityMatrixForNontrivialSolution(int thread_id,
                                                                                   Real k_x,
                                                                                   Real k_y,
                                                                                   Real particle_velocity,
                                                                                   Real coupling_strength,
                                                                                   Real noise,
                                                                                   Real j_1,
                                                                                   const std::vector<MultiprecisionReal> &bessel_functions)
{
  const Real alpha = 0.0;
  const Complex exp_plus_i_alpha = std::exp(kI * alpha), exp_minus_i_alpha = std::exp(-kI * alpha);
  const MultiprecisionReal &I_zero = 1.0 * bessel_functions[0];
  const MultiprecisionReal &I_one = bessel_functions[1];
  const MultiprecisionReal &I_minus_one = 1.0 * bessel_functions[1];

  Eigen::MatrixXcd &eigen_stability_matrix = eigen_stability_matrices_[thread_id];
  Complex matrix_element(0.0, 0.0);
  int nn = 0, mm = 0;
  for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
  {
    nn = max_mode_idx_ + n;
    const MultiprecisionReal &I_n = 1.0 * bessel_functions[std::abs(n)];
    const MultiprecisionReal &I_n_minus_one = 1.0 * bessel_functions[std::abs(n - 1)];
    const MultiprecisionReal &I_n_plus_one = 1.0 * bessel_functions[std::abs(n + 1)];

    for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
    {
      mm = max_mode_idx_ + m;
      matrix_element.real(0.0), matrix_element.imag(0.0);
      if (n - 1 == m)
      {
        matrix_element += 0.5 * particle_velocity * 2.0 * kPi * (kI * k_x - k_y);
        matrix_element += Real(0.5 * n * coupling_strength * I_one / I_zero) * exp_minus_i_alpha;
      }
      if (n + 1 == m)
      {
        matrix_element += 0.5 * particle_velocity * 2.0 * kPi * (kI * k_x + k_y);
        matrix_element -= Real(0.5 * n * coupling_strength * I_minus_one / I_zero) * exp_plus_i_alpha;
      }
      if (n == m)
      {
        matrix_element -= n * n * noise;
      }
      if (1 == m)
      {
        matrix_element += Real(0.5 * n * coupling_strength * j_1 * I_n_minus_one / I_zero) * exp_minus_i_alpha;
      }
      if (-1 == m)
      {
        matrix_element -= Real(0.5 * n * coupling_strength * j_1 * I_n_plus_one / I_zero) * exp_plus_i_alpha;
      }
      if (0 == m)
      {
        matrix_element -=
            Real(0.5 * n * coupling_strength * j_1 * I_one * I_n_minus_one / (I_zero * I_zero)) * exp_minus_i_alpha;
        matrix_element +=
            Real(0.5 * n * coupling_strength * j_1 * I_minus_one * I_n_plus_one / (I_zero * I_zero)) * exp_plus_i_alpha;
//        matrix_element -= Real(n * n / gamma * I_one * I_n / (I_zero * I_zero) * j_1);
      }

      eigen_stability_matrix(nn, mm) = matrix_element;
    } // m
  } // n
}

//void NonhomogeneousPerturbationZeroLag::FillInStabilityMatrixForNontrivialSolutionAgainstZeroWavenumber(Real coupling_strength,
//                                                                                                        Real noise,
//                                                                                                        const std::vector<
//                                                                                                            MultiprecisionReal> &bessel_functions)
//{
//  const Real alpha = 0.0;
//  const Complex exp_plus_i_alpha = std::exp(kI * alpha), exp_minus_i_alpha = std::exp(-kI * alpha);
//
//  const MultiprecisionReal &I_zero = 1.0 * bessel_functions[0];
//  const MultiprecisionReal &I_one = bessel_functions[1];
//  const MultiprecisionReal &I_minus_one = 1.0 * bessel_functions[1];
//
//  const MultiprecisionReal two_pi = boost::math::constants::two_pi<MultiprecisionReal>();
//
//  Complex matrix_element(0.0, 0.0);
//  int nn = 0, mm = 0;
//  for (int n = -max_mode_idx_; n <= max_mode_idx_; ++n)
//  {
//    nn = max_mode_idx_ + n;
//    const MultiprecisionReal &I_n = 1.0 * bessel_functions[std::abs(n)];
//    const MultiprecisionReal &I_n_minus_one = 1.0 * bessel_functions[std::abs(n - 1)];
//    const MultiprecisionReal &I_n_plus_one = 1.0 * bessel_functions[std::abs(n + 1)];
//
//    for (int m = -max_mode_idx_; m <= max_mode_idx_; ++m)
//    {
//      mm = max_mode_idx_ + m;
//      matrix_element.real(0.0), matrix_element.imag(0.0);
//      if (n - 1 == m)
//      {
//        matrix_element += Real(0.5 * n * coupling_strength * I_one / I_zero) * exp_minus_i_alpha;
//      }
//      if (n + 1 == m)
//      {
//        matrix_element -= Real(0.5 * n * coupling_strength * I_minus_one / I_zero) * exp_plus_i_alpha;
//      }
//      if (0 == m)
//      {
//        matrix_element +=
//            Real(0.5 * n * coupling_strength * I_one * I_n_minus_one / (I_zero * I_zero)) * exp_minus_i_alpha;
//        matrix_element -=
//            Real(0.5 * n * coupling_strength * I_minus_one * I_n_plus_one / (I_zero * I_zero)) * exp_plus_i_alpha;
//      }
//      if (1 == m)
//      {
//        matrix_element += Real(0.5 * n * coupling_strength * I_n_minus_one / I_zero) * exp_minus_i_alpha;
//      }
//      if (-1 == m)
//      {
//        matrix_element -= Real(0.5 * n * coupling_strength * I_n_plus_one / I_zero) * exp_plus_i_alpha;
//      }
//      if (n == m)
//      {
//        matrix_element -= n * n * noise;
//      }
//
//      stability_matrix_(nn, mm) = matrix_element;
//    } // m
//  } // n
//}