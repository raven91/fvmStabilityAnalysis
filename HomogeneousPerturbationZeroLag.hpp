//
// Created by Nikita Kruk on 2019-06-23.
//

#ifndef FVMSTABILITYANALYSIS_HOMOGENEOUSPERTURBATIONZEROLAG_HPP
#define FVMSTABILITYANALYSIS_HOMOGENEOUSPERTURBATIONZEROLAG_HPP

#include "Definitions.hpp"

#include <vector>

#include <eigen3/Eigen/Dense>
//#include </home/nkruk/libs/eigen3/include/eigen3/Eigen/Dense>

class HomogeneousPerturbationZeroLag
{
 public:

  HomogeneousPerturbationZeroLag(int max_mode_idx = 50);
  ~HomogeneousPerturbationZeroLag();

  void StabilityOfUniformSolution();
  void StabilityOfNontrivialSolution();

 private:

  int max_mode_idx_;
  Eigen::MatrixXd stability_matrix_;
  std::vector<Real> real_eigenvalues_;

  void CreateOutputFileForUniformSolution(std::ofstream &stability_analysis_file) const;
  void CreateInputFileForNontrivialSolution(std::ifstream &order_parameter_file) const;
  void CreateOutputFileForNontrivialSolution(std::ofstream &stability_analysis_file) const;

};

#endif //FVMSTABILITYANALYSIS_HOMOGENEOUSPERTURBATIONZEROLAG_HPP
