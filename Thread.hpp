//
// Created by Nikita Kruk on 2019-07-06.
//

#ifndef FVMSTABILITYANALYSIS_THREAD_HPP
#define FVMSTABILITYANALYSIS_THREAD_HPP

#include "Definitions.hpp"

class Thread
{
 public:

  Thread(int argc, char **argv);
  virtual ~Thread();

  int GetRank();
  int GetNumberOfMpichThreads();

 protected:

  int root_rank_;
  int rank_;
  int number_of_mpich_threads_;

};

#endif //FVMSTABILITYANALYSIS_THREAD_HPP
