//
// Created by Nikita Kruk on 2019-07-06.
//

#include "Parallelization.hpp"

#if defined(MPI_FOR_PARAMETER_SPAN)
#include <mpi.h>
#endif

#if defined(MPI_FOR_PARAMETER_SPAN)
void LaunchParallelSession(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
}
#else
void LaunchParallelSession(int argc, char **argv)
{

}
#endif

#if defined(MPI_FOR_PARAMETER_SPAN)
void FinalizeParallelSession()
{
  MPI_Finalize();
}
#else
void FinalizeParallelSession()
{

}
#endif