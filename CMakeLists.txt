cmake_minimum_required (VERSION 3.16)
project(SU3-kokkos LANGUAGES C CXX)
set(EXE32 bench_f32)
set(EXE64 bench_f64)
set(CMAKE_CUDA_ARCHITECTURES 70 80)

set(ALIGN OFF CACHE STRING "Align workloads to include device copies")
set(CMAKE_CXX_STANDARD 17)

set(SOURCES su3_nn_bench.cpp su3.hpp lattice.hpp)

if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL not specificied, select either Kokkos or RAJA")
else ()
    message(STATUS "Selected model: ${MODEL}")
endif ()

add_executable (${EXE32} ${SOURCES})
add_executable (${EXE64} ${SOURCES})

if (${MODEL} STREQUAL "Kokkos")
    find_package(Kokkos REQUIRED)
    target_sources(${EXE32} PRIVATE mat_nn_kokkos.hpp)
    target_sources(${EXE64} PRIVATE mat_nn_kokkos.hpp)
    target_link_libraries(${EXE32} Kokkos::kokkos)
    target_link_libraries(${EXE64} Kokkos::kokkos)
    add_compile_definitions(USE_KOKKOS)
elseif (${MODEL} STREQUAL "RAJA")
else ()
    message(FATAL_ERROR "Invalid model, must be either Kokkos or RAJA")
endif ()

if (ALIGN)
    add_compile_definitions(ALIGNED_WORK)
endif()

ADD_COMPILE_DEFINITIONS(USE_KOKKOS)
target_compile_definitions(${EXE32} PUBLIC PRECISION=1)

install(TARGETS ${EXE32} ${EXE64} DESTINATION bin)
