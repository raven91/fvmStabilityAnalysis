//
// Created by Nikita Kruk on 2019-07-06.
//

#include "Thread.hpp"

Thread::Thread(int argc, char **argv)
{
  root_rank_ = 0;
  rank_ = 0;
  number_of_mpich_threads_ = 1;
}

Thread::~Thread()
{

}

int Thread::GetRank()
{
  return rank_;
}

int Thread::GetNumberOfMpichThreads()
{
  return number_of_mpich_threads_;
}