
echo "Building OMP version"
mkdir -p build_omp
cd build_omp
CC=amdclang CXX=amdclang++ cmake -DCMAKE_BUILD_TYPE=Release -DMODEL=OMP -DTARGET=AMD -DALIGN=On -DARCH=gfx906 ../
make
cd ..


echo "Building HIP version"
mkdir -p build_hip
cd build_hip
CC=hipcc CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -DMODEL=HIP -DALIGN=On ../
make
cd ..

echo "Building RAJA version"
mkdir -p build_raja
cd build_raja
CC=amdclang CXX=amdclang++ cmake -DCMAKE_BUILD_TYPE=Release -DMODEL=RAJA -DTARGET=HIP -DALIGN=On -DBLT_DIR=$(spack location -i blt)  -DCMAKE_CXX_FLAGS="-march=native" ../
make
cd ..


echo "Building KOKKOS version"
mkdir -p build_kokkos
cd build_kokkos
CC=hipcc CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -DMODEL=Kokkos -DTARGET=HIP -DALIGN=On -DCMAKE_CXX_FLAGS="-march=native" ../
make
cd ..

echo "Building SYCL version"
mkdir -p build_sycl
cd build_sycl
CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release -DMODEL=SYCL -DALIGN=On -DARCH=gfx906 -DSYCL_TARGET=amdgcn-amd-amdhsa  -DSYCL_COMPILER_DIR=$(spack location -i dpcpp) -DCMAKE_CXX_FLAGS="-march=native" ../
make
cd ..

