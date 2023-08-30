cmake_minimum_required (VERSION 3.16)
project(SU3-kkokkos
  LANGUAGES C CXX
  )

set(EXE32 bench_f32)
set(EXE64 bench_f64)
set(CMAKE_CUDA_ARCHITECTURES 70 80)

set(CMAKE_CXX_STANDARD 17)

find_package(Kokkos REQUIRED)

set(SOURCES su3_nn_bench.cpp su3.hpp lattice.hpp mat_nn_kokkos.hpp)

add_executable (${EXE32} ${SOURCES})
add_executable (${EXE64} ${SOURCES})

TARGET_LINK_LIBRARIES(${EXE32} Kokkos::kokkos)
TARGET_LINK_LIBRARIES(${EXE64} Kokkos::kokkos)

ADD_COMPILE_DEFINITIONS(USE_KOKKOS)
target_compile_definitions(${EXE32} PUBLIC PRECISION=1)

