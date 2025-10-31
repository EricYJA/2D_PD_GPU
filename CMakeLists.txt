cmake_minimum_required(VERSION 3.18)
project(2D_Hassa_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(BUILD_CPU "Build CPU-only executable" ON)
option(BUILD_GPU "Build GPU-enabled executable (requires CUDA)" ON)
option(BUILD_SEQ "Build sequential executable" ON)

# Find core packages. Prefer modern imported targets where available.
find_package(MPI REQUIRED)

# Try to find Eigen (Config-mode preferred, fall back to NO_MODULE if needed)
find_package(Eigen3 3.3 QUIET)
if(NOT TARGET Eigen3::Eigen)
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# CUDA is optional; only required if BUILD_GPU is ON
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(HAVE_CUDA TRUE)
else()
  set(HAVE_CUDA FALSE)
endif()

# Locate ParMETIS / METIS if available. Do not hard-fail here; only require when
# creating targets that need them.
find_library(PARMETIS_LIB NAMES parmetis)
find_library(METIS_LIB NAMES metis)
if(NOT PARMETIS_LIB)
  message(STATUS "ParMETIS not found: parallel partitioning may be unavailable.")
endif()
if(NOT METIS_LIB)
  message(STATUS "METIS not found: graph partitioning may be unavailable.")
endif()

# Helper variables
set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(COMMON_SRCS
  src/SetupParticleSystem.cpp
  src/matrix.cpp
)

# Choose MPI imported target if available
if(TARGET MPI::MPI_CXX)
  set(MPI_TARGET MPI::MPI_CXX)
elseif(TARGET MPI::MPI_C)
  set(MPI_TARGET MPI::MPI_C)
else()
  # fallback to legacy variables if necessary
  if(MPI_CXX_COMPILE_FLAGS)
    message(WARNING "MPI imported target not available; falling back to legacy MPI variables.")
    set(MPI_TARGET "${MPI_LIBRARIES}")
  endif()
endif()

# Function to append optional libs if they exist
function(maybe_link_vars target)
  if(PARMETIS_LIB)
    target_link_libraries(${target} PRIVATE ${PARMETIS_LIB})
  endif()
  if(METIS_LIB)
    target_link_libraries(${target} PRIVATE ${METIS_LIB})
  endif()
endfunction()

# CPU-only target
if(BUILD_CPU)
  add_executable(2D_Hassa_test_cpu
    src/main.cpp
    ${COMMON_SRCS}
    src/mechanics.cpp
  )
  target_include_directories(2D_Hassa_test_cpu PRIVATE ${PROJECT_INCLUDE_DIR})
  # Prefer modern targets
  if(TARGET Eigen3::Eigen)
    target_link_libraries(2D_Hassa_test_cpu PRIVATE Eigen3::Eigen)
  endif()
  if(MPI_TARGET)
    target_link_libraries(2D_Hassa_test_cpu PRIVATE ${MPI_TARGET})
  endif()
  maybe_link_vars(2D_Hassa_test_cpu)
endif()

# GPU-enabled target (only if CUDA found)
if(BUILD_GPU)
  if(HAVE_CUDA)
    add_executable(2D_Hassa_test_gpu
      src/main.cu
      ${COMMON_SRCS}
    )
    target_include_directories(2D_Hassa_test_gpu PRIVATE ${PROJECT_INCLUDE_DIR})
    if(TARGET Eigen3::Eigen)
      target_link_libraries(2D_Hassa_test_gpu PRIVATE Eigen3::Eigen)
    endif()
    if(MPI_TARGET)
      target_link_libraries(2D_Hassa_test_gpu PRIVATE ${MPI_TARGET})
    endif()
    # Link CUDA runtime via CUDAToolkit imported target if present
    if(TARGET CUDA::cudart)
      target_link_libraries(2D_Hassa_test_gpu PRIVATE CUDA::cudart)
    elseif(TARGET CUDAToolkit::cudart)
      target_link_libraries(2D_Hassa_test_gpu PRIVATE CUDAToolkit::cudart)
    endif()
    maybe_link_vars(2D_Hassa_test_gpu)
  else()
    message(WARNING "BUILD_GPU is ON but CUDA was not found. GPU executable will not be created.")
  endif()
endif()

# Sequential version
if(BUILD_SEQ)
  add_executable(main_seq
    src/main_seq.cpp
    ${COMMON_SRCS}
  )
  target_include_directories(main_seq PRIVATE ${PROJECT_INCLUDE_DIR})
  if(TARGET Eigen3::Eigen)
    target_link_libraries(main_seq PRIVATE Eigen3::Eigen)
  endif()
  if(MPI_TARGET)
    target_link_libraries(main_seq PRIVATE ${MPI_TARGET})
  endif()
  maybe_link_vars(main_seq)
endif()

message(STATUS "Configuration summary:")
message(STATUS "  BUILD_CPU: ${BUILD_CPU}")
message(STATUS "  BUILD_GPU: ${BUILD_GPU}")
message(STATUS "  BUILD_SEQ: ${BUILD_SEQ}")
message(STATUS "  HAVE_CUDA: ${HAVE_CUDA}")
message(STATUS "  Eigen target: ${TARGET_EIGEN_PRESENT}")