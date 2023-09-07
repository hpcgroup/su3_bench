cmake_minimum_required (VERSION 3.16)

project(SU3-kkokkos
  LANGUAGES C CXX
  )

set(EXE32 bench_f32)
set(EXE64 bench_f64)
set(CMAKE_CUDA_ARCHITECTURES 70)

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
    target_sources(${EXE32} PRIVATE mat_nn_kokkos.hpp)
    target_sources(${EXE64} PRIVATE mat_nn_kokkos.hpp)

    find_package(Kokkos REQUIRED)
    target_link_libraries(${EXE32} Kokkos::kokkos)
    target_link_libraries(${EXE64} Kokkos::kokkos)
    add_compile_definitions(USE_KOKKOS)

elseif (${MODEL} STREQUAL "RAJA")
    target_sources(${EXE32} PRIVATE mat_nn_raja.hpp)
    target_sources(${EXE64} PRIVATE mat_nn_raja.hpp)

    if (${BACKEND} STREQUAL "CUDA")
       enable_language(CUDA)
       set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -forward-unknown-to-host-compiler -arch=sm_70 -extended-lambda --expt-relaxed-constexpr -use_fast_math -restrict -keep")
       set_source_files_properties(mat_nn_raja.cpp PROPERTIES LANGUAGE CUDA)
       set_source_files_properties(su3_nn_bench.cpp PROPERTIES LANGUAGE CUDA)
    endif ()

    find_package(Threads REQUIRED)
    find_package(RAJA REQUIRED)
    find_package(umpire REQUIRED)
    find_package(chai REQUIRED)


    # HACK: this is ugly, but it is a hack. It assumes that
    # there is a symlink in the source called libcuda_runtime.so and
    # points to some library
    target_link_directories(${EXE32} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${EXE32} PUBLIC RAJA chai)
    set_property(TARGET ${EXE32} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    # HACK: this is ugly, but it is a hack. It assumes that
    # there is a symlink in the source called libcuda_runtime.so and
    # points to some library
    target_link_directories(${EXE64} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${EXE64} PUBLIC RAJA chai)
    set_property(TARGET ${EXE64} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    add_compile_definitions(USE_RAJA)

    else ()
       message(FATAL_ERROR "Invalid model, must be either Kokkos or RAJA")
    endif ()

target_compile_definitions(${EXE32} PUBLIC PRECISION=1)

install(TARGETS ${EXE32} ${EXE64} DESTINATION bin)
