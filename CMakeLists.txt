cmake_minimum_required (VERSION 3.16)

project(SU3BENCH
  LANGUAGES C CXX
)

set(EXE32 bench_f32)
set(EXE64 bench_f64)

set(CMAKE_CXX_STANDARD 17)

set(SOURCES su3_nn_bench.cpp su3.hpp lattice.hpp)

if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL not specificied, select either Kokkos or RAJA")
else ()
    message(STATUS "Selected model: ${MODEL}")
endif ()

if (${MODEL} STREQUAL "Kokkos")
    add_executable (${EXE32} ${SOURCES})
    add_executable (${EXE64} ${SOURCES})

    target_sources(${EXE32} PRIVATE mat_nn_kokkos.hpp)
    target_sources(${EXE64} PRIVATE mat_nn_kokkos.hpp)
    
    find_package(Kokkos REQUIRED)
    target_link_libraries(${EXE32} Kokkos::kokkos)
    target_link_libraries(${EXE64} Kokkos::kokkos)

    target_compile_definitions(${EXE32} PUBLIC PRECISION=1)
    add_compile_definitions(USE_KOKKOS)
elseif (${MODEL} STREQUAL "RAJA")
    list (APPEND SOURCES mat_nn_raja.hpp)
    
    find_package(RAJA REQUIRED)
    find_package(umpire REQUIRED)
    find_package(chai REQUIRED)

    set (DEPENDS RAJA chai umpire)

    if (ENABLE_OPENMP)
        list (APPEND DEPENDS openmp)
    endif ()

    if (ENABLE_CUDA)
        list (APPEND DEPENDS cuda)
    endif ()

    if (ENABLE_HIP)
        list (APPEND DEPENDS blt::hip)
        list (APPEND DEPENDS blt::hip_runtime)
    endif ()

    if (ENABLE_SYCL)
        list (APPEND DEPENDS sycl)
    endif ()
    
    include("${BLT_SOURCE_DIR}/SetupBLT.cmake")

    blt_add_executable(
        NAME ${EXE32}
        SOURCES ${SOURCES}
        DEFINES USE_RAJA PRECISION=1
        DEPENDS_ON ${DEPENDS}
    )

    blt_add_executable(
        NAME ${EXE64}
        SOURCES ${SOURCES}
        DEFINES USE_RAJA
        DEPENDS_ON ${DEPENDS}
    )
else ()
    message(FATAL_ERROR "Invalid model, must be either Kokkos or RAJA")
endif ()

install(TARGETS ${EXE32} ${EXE64} DESTINATION bin)
