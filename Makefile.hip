ALIGNED ?= no

#
HIP_PLATFORM=$(shell hipconfig --platform)
ifeq (${HIP_PLATFORM}, $(filter $(HIP_PLATFORM), nvcc nvidia))
  # ARCH = V100 | A100
  ifeq ($(ARCH),A100)
    HIPCC_FLAGS = -gencode arch=compute_80,code=sm_80
  else
    HIPCC_FLAGS = -gencode arch=compute_70,code=sm_70
  endif
else
  # ARCH = MI60 | MI100
  ifeq ($(ARCH),MI100)
    HIPCC_FLAGS = --offload-arch=gfx908
  else ifeq ($(ARCH),MI250)
    HIPCC_FLAGS = --offload-arch=gfx90a
  else
    HIPCC_FLAGS = --offload-arch=gfx906
  endif
endif

# USE MILC Complex data types
#HIPCC_FLAGS += -DMILC_COMPLEX

ifeq ($(ALIGNED),yes)
  DEFINES = -DALIGNED_WORK
endif

CC = hipcc
CFLAGS += -O3 $(HIPCC_FLAGS)
INCLUDES = -DUSE_HIP #-DMILC_COMPLEX
DEPENDS = su3.hpp lattice.hpp mat_nn_hip.hpp

bench_f32_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) $(DEFINES) -DPRECISION=1 -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) $(DEFINES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_hip.exe bench_f32_hip.exe

clean:
	rm -f *.exe
