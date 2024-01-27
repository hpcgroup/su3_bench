echo "Building RAJA version"
mkdir -p build_nvidia_raja
cd build_nvidia_raja
CC=gcc cmake -DCMAKE_BUILD_TYPE=Release -DMODEL=RAJA -DTARGET=CUDA -DBLT_DIR=$(spack location -i blt) -DCMAKE_CUDA_ARCHITECTURES="70" ../
make
cd ..

echo "Building KOKKOS version"
mkdir -p build_nvidia_kokkos
cd build_nvidia_kokkos
cmake -DCMAKE_BUILD_TYPE=Release -DMODEL=Kokkos -DALIGN=On ../
make
cd ..

