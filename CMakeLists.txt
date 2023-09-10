cmake_minimum_required (VERSION 3.16)
project(SU3-kokkos
  LANGUAGES C CXX
  )

set(CMAKE_CXX_STANDARD 17)

SET(EXE32 bench_f32_kokkos.exe)
SET(EXE64 bench_f64_kokkos.exe)
set (CMAKE_CUDA_ARCHITECTURES 70 80)

set(ALIGN ON CACHE STRING "Align workloads to include device copies")

find_package(Kokkos REQUIRED)

set(SOURCES su3_nn_bench.cpp su3.hpp lattice.hpp mat_nn_kokkos.hpp)

add_executable (${EXE32} ${SOURCES})
add_executable (${EXE64} ${SOURCES})

TARGET_LINK_LIBRARIES(${EXE32} Kokkos::kokkos)
TARGET_LINK_LIBRARIES(${EXE64} Kokkos::kokkos)

if (ALIGN)
    add_compile_definitions(ALIGNED_WORK)
endif()

ADD_COMPILE_DEFINITIONS(USE_KOKKOS)
target_compile_definitions(${EXE32} PUBLIC PRECISION=1)

install(TARGETS ${EXE32} ${EXE64} DESTINATION bin)
