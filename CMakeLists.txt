cmake_minimum_required (VERSION 3.16)
project(SU3-kokkos LANGUAGES C CXX)
set(EXE32 bench_f32)
set(EXE64 bench_f64)

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
    #add_executable (${EXE32} ${SOURCES})
    #add_executable (${EXE64} ${SOURCES})

    target_sources(${EXE32} PRIVATE mat_nn_kokkos.hpp)
    target_sources(${EXE64} PRIVATE mat_nn_kokkos.hpp)

    find_package(Kokkos REQUIRED)
    target_link_libraries(${EXE32} Kokkos::kokkos)
    target_link_libraries(${EXE64} Kokkos::kokkos)
    target_compile_definitions(${EXE32} PUBLIC PRECISION=1)
    add_compile_definitions(USE_KOKKOS)

elseif (${MODEL} STREQUAL "RAJA")
    find_package(Threads REQUIRED)
    find_package(RAJA REQUIRED)
    find_package(umpire REQUIRED)
    find_package(chai REQUIRED)

    if (${BACKEND} STREQUAL "CUDA")
       set(ENABLE_CUDA ON)
       enable_language(CUDA)
       find_package(CUDA REQUIRED)
       set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -forward-unknown-to-host-compiler -arch=${CUDA_ARCH} -extended-lambda --expt-relaxed-constexpr -restrict -keep -DTHRUST_IGNORE_CUB_VERSION_CHECK ${CUDA_EXTRA_FLAGS}")
       set_source_files_properties(mat_nn_raja.cpp PROPERTIES LANGUAGE CUDA)
       set_source_files_properties(su3_nn_bench.cpp PROPERTIES LANGUAGE CUDA)
       set_property(TARGET ${EXE32} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
       set_property(TARGET ${EXE64} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
       # HACK: this is ugly, but it is a hack. It assumes that
       # there is a symlink in the source called libcuda_runtime.so and
       # points to some library
       target_link_directories(${EXE32} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
       target_link_directories(${EXE64} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    elseif (${BACKEND} STREQUAL "HIP")
      set(ENABLE_HIP ON)
      find_package(hip REQUIRED)
    endif ()

    include(${BLT_DIR}/SetupBLT.cmake)

    target_link_libraries(${EXE32} PUBLIC RAJA)
    target_link_libraries(${EXE64} PUBLIC RAJA)
    target_link_libraries(${EXE32} PUBLIC chai)
    target_link_libraries(${EXE64} PUBLIC chai)
    target_link_libraries(${EXE32} PUBLIC umpire)
    target_link_libraries(${EXE64} PUBLIC umpire)
    add_compile_definitions(USE_RAJA)

else ()
    message(FATAL_ERROR "Invalid model, must be either Kokkos or RAJA")
endif ()

if (ALIGN)
    add_compile_definitions(ALIGNED_WORK)
endif()

ADD_COMPILE_DEFINITIONS(USE_KOKKOS)
target_compile_definitions(${EXE32} PUBLIC PRECISION=1)

install(TARGETS ${EXE32} ${EXE64} DESTINATION bin)
