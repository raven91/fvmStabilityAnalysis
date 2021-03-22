//
// Created by Nikita Kruk on 2019-07-06.
//

#ifndef FVMSTABILITYANALYSIS_THREADFORPARAMETERSPAN_HPP
#define FVMSTABILITYANALYSIS_THREADFORPARAMETERSPAN_HPP

#include "Thread.hpp"

class ThreadForParameterSpan : public Thread
{
 public:

  ThreadForParameterSpan(int argc, char **argv);
  ~ThreadForParameterSpan();

 private:

};

#endif //FVMSTABILITYANALYSIS_THREADFORPARAMETERSPAN_HPP
