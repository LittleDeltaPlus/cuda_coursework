cmake_minimum_required(VERSION 3.17)
project(Cuda_Gaussian LANGUAGES CUDA CXX)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcudadevrt")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")



add_executable(Cuda_Gaussian main.cu)

set_target_properties(
        Cuda_Gaussian
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)