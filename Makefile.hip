#
HIP_PLATFORM=$(shell hipconfig --compiler)
ifeq (${HIP_PLATFORM}, nvcc)
	HIPCC_FLAGS = -gencode=arch=compute_70,code=sm_70 
endif
ifeq (${HIP_PLATFORM}, hcc)
	HIPCC_FLAGS = -Wno-deprecated-register
endif

LAT_CHECK = false

CC = hipcc $(HIPCC_FLAGS)
CFLAGS = --std c++14 
INCLUDES = -DUSE_HIP
ifeq ($(LAT_CHECK),true)
 INCLUDES += -DLAT_CHECK
endif
DEPENDS = su3.hpp lattice.hpp mat_nn_hip.hpp

bench_f32_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -DPRECISION=1 -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_hip.exe bench_f32_hip.exe

clean:
	rm -f *.exe