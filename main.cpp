#include "HomogeneousPerturbationZeroLag.hpp"
#include "HomogeneousPerturbationNonzeroLag.hpp"
#include "NonhomogeneousPerturbationZeroLag.hpp"
#include "NonhomogeneousPerturbationNonzeroLag.hpp"
#include "NonhomogeneousPerturbationNonzeroLagInterleaved.hpp"
#include "Parallelization.hpp"
#include "Thread.hpp"
#include "ThreadForParameterSpan.hpp"

#include <string> // std::stoi

int kRank = 0;
int kBesselThreads = 2;
int kWaveNumberThreads = 2;

int main(int argc, char **argv)
{
  if (argc > 1)
  {
    kRank = std::stoi(argv[1]);
  }
  if (argc > 2)
  {
    kBesselThreads = std::stoi(argv[2]);
  }
  if (argc > 3)
  {
    kWaveNumberThreads = std::stoi(argv[3]);
  }

  LaunchParallelSession(argc, argv);
  {
#if defined(MPI_FOR_PARAMETER_SPAN)
    ThreadForParameterSpan thread(argc, argv);
#else
    Thread thread(argc, argv);
#endif

    MultiprecisionReal::default_precision(1500); // 1500 used for PRL paper

//    HomogeneousPerturbationZeroLag stability_engine;
//    stability_engine.StabilityOfUniformSolution();
//    stability_engine.StabilityOfNontrivialSolution();

//    HomogeneousPerturbationNonzeroLag stability_engine(100);
//    stability_engine.StabilityOfUniformSolution();
//    stability_engine.StabilityOfNontrivialSolution();

//    NonhomogeneousPerturbationZeroLag stability_engine(true);
//    stability_engine.StabilityOfNontrivialSolution();

    NonhomogeneousPerturbationNonzeroLag stability_engine(&thread, false);
//    stability_engine.StabilityOfUniformSolution();
    stability_engine.StabilityOfNontrivialSolution();

//    NonhomogeneousPerturbationNonzeroLagInterleaved stability_engine(true);
//    stability_engine.StabilityOfNontrivialSolution();
  }
  FinalizeParallelSession();

  return 0;
}