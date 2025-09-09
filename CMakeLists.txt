cmake_minimum_required(VERSION 3.18)
project(2D_Hassa_test LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

include_directories(
  ${MPI_INCLUDE_PATH}
  ${EIGEN3_INCLUDE_DIRS}
  include
)

find_library(PARMETIS_LIB NAMES parmetis REQUIRED)
find_library(METIS_LIB   NAMES metis   REQUIRED)

# CPU‐only version
add_executable(2D_Hassa_test_cpu
  src/main.cpp
  src/SetupParticleSystem.cpp
  src/matrix.cpp
  src/mechanics.cpp
)
target_link_libraries(2D_Hassa_test_cpu
  Eigen3::Eigen
  ${MPI_LIBRARIES}
  ${PARMETIS_LIB}
  ${METIS_LIB}
)

# GPU‐enabled version
add_executable(2D_Hassa_test_gpu
  src/main.cu           # mirror of main.cpp, with CUDA calls
  src/SetupParticleSystem.cpp
  src/matrix.cpp  # CPU helpers still needed
)
target_link_libraries(2D_Hassa_test_gpu
  Eigen3::Eigen
  ${MPI_LIBRARIES}
  ${PARMETIS_LIB}
  ${METIS_LIB}
  CUDA::cudart
)

# Sequential version
add_executable(main_seq
  src/main_seq.cpp
  src/SetupParticleSystem.cpp
  src/matrix.cpp
)
target_link_libraries(main_seq
  Eigen3::Eigen
  ${MPI_LIBRARIES}
  ${PARMETIS_LIB}
  ${METIS_LIB}
)