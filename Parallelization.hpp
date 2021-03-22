//
// Created by Nikita Kruk on 2019-07-06.
//

#ifndef FVMSTABILITYANALYSIS_PARALLELIZATION_HPP
#define FVMSTABILITYANALYSIS_PARALLELIZATION_HPP

#include "Definitions.hpp"

void LaunchParallelSession(int argc, char **argv);
void FinalizeParallelSession();

#endif //FVMSTABILITYANALYSIS_PARALLELIZATION_HPP
