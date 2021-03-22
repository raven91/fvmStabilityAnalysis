//
// Created by Nikita Kruk on 2019-07-06.
//

#include "ThreadForParameterSpan.hpp"

#include <mpi.h>

ThreadForParameterSpan::ThreadForParameterSpan(int argc, char **argv) :
    Thread(argc, argv)
{
  root_rank_ = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_mpich_threads_);
}

ThreadForParameterSpan::~ThreadForParameterSpan()
{

}