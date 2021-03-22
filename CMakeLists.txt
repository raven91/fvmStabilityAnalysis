cmake_minimum_required(VERSION 3.14)
project(fvmStabilityAnalysis)

set(CMAKE_CXX_STANDARD 17)

include_directories(/Users/nikita/Documents/mpich-install/include/
        /usr/local/include/
        /opt/intel/compilers_and_libraries_2019.4.233/mac/mkl/include/
        /opt/OpenBLAS/include/)
link_directories(/Users/nikita/Documents/mpich-install/lib/
        /usr/local/lib/
        /opt/intel/compilers_and_libraries_2019.4.233/mac/mkl/lib/
        /opt/OpenBLAS/lib/)

set(GCC_COVERAGE_COMPILE_FLAGS "-march=native -O3 -flto -DMKL_LP64 -m64")
set(GCC_COVERAGE_LINK_FLAGS "-lmpc -lmpfr -lgmp \
                            -lopenblas \
                            -L/opt/intel/compilers_and_libraries_2019.4.233/mac/mkl/lib -Wl,-rpath,/opt/intel/compilers_and_libraries_2019.4.233/mac/mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
                            -lpthread \
                            -larmadillo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)

add_executable(fvmStabilityAnalysis main.cpp
        HomogeneousPerturbationZeroLag.cpp HomogeneousPerturbationZeroLag.hpp
        Definitions.hpp Definitions.cpp
        HomogeneousPerturbationNonzeroLag.cpp HomogeneousPerturbationNonzeroLag.hpp
        NonhomogeneousPerturbationZeroLag.cpp NonhomogeneousPerturbationZeroLag.hpp
        NonhomogeneousPerturbationNonzeroLag.cpp NonhomogeneousPerturbationNonzeroLag.hpp
        Thread.cpp Thread.hpp
#        ThreadForParameterSpan.cpp ThreadForParameterSpan.hpp
        Parallelization.hpp Parallelization.cpp
        NonhomogeneousPerturbationNonzeroLagInterleaved.cpp NonhomogeneousPerturbationNonzeroLagInterleaved.hpp)