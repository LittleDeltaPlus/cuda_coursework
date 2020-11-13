cmake_minimum_required(VERSION 3.17)
project(OMP_Gaussian)
find_package(OpenMP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")



add_executable(OMP_Gaussian linear.cpp)
# OR
#add_executable(OMP_Gaussian parallel.cpp)
target_link_libraries(OMP_Gaussian PUBLIC OpenMP::OpenMP_CXX)