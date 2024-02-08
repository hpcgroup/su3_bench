cmake_minimum_required (VERSION 3.21)
project(SU3 LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCES su3_nn_bench.cpp)

add_executable(bench_f32 ${SOURCES})
add_executable(bench_f64 ${SOURCES})

target_compile_definitions(bench_f32 PUBLIC PRECISION=1)

if (${MODEL} STREQUAL "CUDA")
    enable_language(CUDA)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    add_compile_definitions(USE_CUDA)
elseif (${MODEL} STREQUAL "HIP")
    enable_language(HIP)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)
    add_compile_definitions(USE_HIP)
elseif (${MODEL} STREQUAL "Kokkos")
    set(CMAKE_CXX_STANDARD 17)

    find_package(Kokkos REQUIRED)

    add_compile_definitions(USE_KOKKOS)
    target_link_libraries(bench_f32 Kokkos::kokkos)
    target_link_libraries(bench_f64 Kokkos::kokkos)
elseif (${MODEL} STREQUAL "RAJA")
    set(CMAKE_CXX_STANDARD 14)

    find_package(RAJA REQUIRED)
    find_package(Umpire REQUIRED)

    add_compile_definitions(USE_RAJA)
    target_link_libraries(bench_f32 RAJA umpire)
    target_link_libraries(bench_f64 RAJA umpire)

    if (${RAJA_BACK_END} STREQUAL "CUDA")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        # TODO Test if I can remove
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

        set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -extended-lambda --expt-relaxed-constexpr --restrict --keep")
    elseif (${RAJA_BACK_END} STREQUAL "HIP")
        enable_language(HIP)
        find_package(hip REQUIRED)
        set(CMAKE_HIP_STANDARD 14)
        # TODO Test if I can remove
        set(CMAKE_HIP_SEPARABLE_COMPILATION ON)

        set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)
    endif ()
elseif (${MODEL} STREQUAL "SYCL")
    set(CMAKE_CXX_STANDARD 17)
    add_compile_definitions(USE_SYCL MILC_COMPLEX)

    separate_arguments(OFFLOAD_FLAGS)

    target_compile_options(bench_f32 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
    target_compile_options(bench_f64 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")

    target_link_options(bench_f32 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
    target_link_options(bench_f64 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
elseif (${MODEL} STREQUAL "OpenACC")
    add_compile_definitions(USE_OPENACC)

    separate_arguments(OFFLOAD_FLAGS)

    target_compile_options(bench_f32 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
    target_compile_options(bench_f64 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")

    target_link_options(bench_f32 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
    target_link_options(bench_f64 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
elseif (${MODEL} STREQUAL "OpenMP-Offload")
    add_compile_definitions(USE_OPENMP)

    separate_arguments(OFFLOAD_FLAGS)

    target_compile_options(bench_f32 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
    target_compile_options(bench_f64 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")

    target_link_options(bench_f32 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
    target_link_options(bench_f64 PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
else () 
    message(FATAL_ERROR "Invalid Model")
endif ()

install(TARGETS bench_f32 bench_f64 DESTINATION bin)