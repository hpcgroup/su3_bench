cmake_minimum_required (VERSION 3.16)
project(SU3-kokkos LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS On)

set(ALIGN OFF CACHE STRING "Align workloads to include device copies")
set(CMAKE_CXX_STANDARD 17)

set(SOURCES su3_nn_bench.cpp su3.hpp lattice.hpp)

macro(register_flag_required NAME DESCRIPTION)
    list(APPEND CUSTOM_FLAGS_TRIPLE "${NAME}" "${DESCRIPTION}" ON "")
endmacro()


if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL not specificied, select either Kokkos or RAJA")
else ()
    message(STATUS "Selected model: ${MODEL}")
endif ()


if (${MODEL} STREQUAL "Kokkos")
    set(EXE32 bench_f32_kokkos)
    set(EXE64 bench_f64_kokkos)

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
    set(EXE32 bench_f32_raja)
    set(EXE64 bench_f64_raja)
    add_executable (${EXE32} ${SOURCES})
    add_executable (${EXE64} ${SOURCES})

    find_package(Threads REQUIRED)
    find_package(RAJA REQUIRED)
    find_package(umpire REQUIRED)
    find_package(chai REQUIRED)
    set(ENABLE_GTEST OFF)
    set(ENABLE_GMOCK OFF)
    set(ENABLE_FRUIT OFF)
    set(ENABLE_GBENCHMARK OFF)
    set(ENABLE_TESTS OFF)

    if (${TARGET} STREQUAL "CUDA")
       set(ENABLE_CUDA ON)
       include(${BLT_DIR}/SetupBLT.cmake)
       set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -forward-unknown-to-host-compiler -extended-lambda --expt-relaxed-constexpr -restrict -keep -DTHRUST_IGNORE_CUB_VERSION_CHECK")
       set_source_files_properties(su3_nn_bench.cpp PROPERTIES LANGUAGE CUDA)
       set_target_properties(${EXE64} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
       set_target_properties(${EXE32} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
       # HACK: this is ugly, but it is a hack. It assumes that
       # there is a symlink in the source called libcuda_runtime.so and
       # points to some library
       target_link_directories(${EXE32} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
       target_link_directories(${EXE64} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
     elseif (${TARGET} STREQUAL "HIP")
      set(ENABLE_HIP ON)
       include(${BLT_DIR}/SetupBLT.cmake)
      find_package(hip REQUIRED)
    endif ()


    target_link_libraries(${EXE32} PUBLIC RAJA)
    target_link_libraries(${EXE64} PUBLIC RAJA)
    target_link_libraries(${EXE32} PUBLIC chai)
    target_link_libraries(${EXE64} PUBLIC chai)
    target_link_libraries(${EXE32} PUBLIC umpire)
    target_link_libraries(${EXE64} PUBLIC umpire)
    add_compile_definitions(USE_RAJA)
    if (KERNEL)
      add_compile_definitions(__RAJA_KERNEL__)
    endif()
elseif (${MODEL} STREQUAL "HIP")
    set(EXE32 bench_f32_hip)
    set(EXE64 bench_f64_hip)
    add_executable (${EXE32} ${SOURCES})
    add_executable (${EXE64} ${SOURCES})
		add_compile_definitions(USE_HIP)

		find_package(HIP REQUIRED)
		find_package(rocthrust REQUIRED)

		target_include_directories(${EXE32} PUBLIC ${rocthrust_INCLUDE_DIR})
		target_include_directories(${EXE64} PUBLIC ${rocthrust_INCLUDE_DIR})
elseif (${MODEL} STREQUAL "OMP")
    set(EXE32 bench_f32_omp)
    set(EXE64 bench_f64_omp)
    add_executable (${EXE32} ${SOURCES})
    add_executable (${EXE64} ${SOURCES})

    add_compile_definitions(USE_OPENMP)

    if(NOT DEFINED ARCH)
      message(FATAL_ERROR "Specify device architecture -DARCH=")
    endif()
    if (NOT DEFINED TARGET)
      message(FATAL_ERROR "Specify device architecture -DTARGET=AMD|NVIDIA")
    endif()

    if ("${TARGET}" STREQUAL "NVIDIA")
      set(OMP_FLAGS "-fopenmp -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=${ARCH}")
    elseif ("${TARGET}" STREQUAL "AMD")
      set(OMP_FLAGS "-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa  -march=${ARCH}")
    else()
      message(FATAL_ERROR "Please specify the TARGET (AMD or NVIDIA)")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMP_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OMP_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -foffload-lto")
else ()
				message(FATAL_ERROR "Invalid model, must be: Kokkos, RAJA or HIP")
endif ()

if (ALIGN)
    add_compile_definitions(ALIGNED_WORK)
endif()



#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_compile_definitions(${EXE32} PUBLIC PRECISION=1)

install(TARGETS ${EXE32} ${EXE64} DESTINATION bin)
