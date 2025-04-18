cmake_minimum_required(VERSION 3.10)
project(2D_Hassa_test CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_library(PARMETIS_LIB NAMES parmetis REQUIRED)
find_library(METIS_LIB NAMES metis REQUIRED)

include_directories(include)

add_executable(2D_Hassa_test src/main.cpp src/SetupParticleSystem.cpp src/matrix.cpp src/mechanics.cpp)

target_link_libraries(2D_Hassa_test 
    Eigen3::Eigen 
    ${MPI_LIBRARIES}
    ${PARMETIS_LIB}
    ${METIS_LIB}
)