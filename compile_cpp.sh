#!/bin/bash

#COMPILATION
#-----------
#- Some parts of the code are written in C++ electron_density_opt_omp.cpp and overlap_matrix.cpp.

#To compile overlap_matrix.cpp
c++ -O3 -Wall -shared -std=c++17 -fPIC  -fopenmp $(python3 -m pybind11 --includes) overlap_matrix.cpp -o overlap_matrix$(python3-config --extension-suffix)

#Output: overlap_matrix.cpython-310-x86_64-linux-gnu.so

c++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp $(python3 -m pybind11 --includes) electron_density_opt_omp.cpp -o electron_density_opt_omp$(python3-config --extension-suffix)
#Output: electron_density_opt_omp.cpython-310-x86_64-linux-gnu.so


