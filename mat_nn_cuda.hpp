// Cuda implementation
#include <cuda_runtime.h>

#define CUCHECK(err, s) \
  if (err != cudaSuccess) { \
        printf("%s (error code %d:%s)!\n", s, err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
  }

#define THREADS_PER_SITE 36

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	

__global__ void k_mat_nn(const site * __restrict__ a, const su3_matrix * __restrict__ b, site * __restrict__ c) {
  const int i = blockIdx.x;

  for (int j = threadIdx.x; j < 4; j += blockDim.x) {
    for (int k = threadIdx.y; k < 3; k += blockDim.y) {
      for (int l = threadIdx.z; l < 3; l += blockDim.z) {
        Complx cc = {0.0, 0.0};
        for (int m = 0; m < 3; m++) {
          cc += a[i].link[j].e[k][m] * b[j].e[m][l];
        }
        c[i].link[j].e[k][l] = cc; 
      }
    }
  }
}

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
		  size_t total_sites, size_t iterations, size_t threadsPerBlock, int use_device, Profile *profile)
{
  int size_a = sizeof(site) * total_sites;
  int size_b = sizeof(su3_matrix) * 4;
  int size_c = sizeof(site) * total_sites;

  if (threadsPerBlock == 0)
    threadsPerBlock = THREADS_PER_SITE;

  // Device initialization
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("ERROR: No devices found\n");
    exit(1);
  }

  struct cudaDeviceProp device_prop;
  if (verbose >= 3) {
    for (int i = 0; i < deviceCount; ++i) {
      cudaGetDeviceProperties(&device_prop, i);
      printf("Located device %d: %s\n", i, device_prop.name);
    }
  }
  if (use_device == -1)
    use_device = 0;
  else if (use_device >= deviceCount) {
    printf("ERROR: Device %d not found\n", use_device);
    exit(1);
  }
  cudaSetDevice(use_device);
  if (verbose >= 2) {
    cudaGetDeviceProperties(&device_prop, use_device);
    printf("Using device %d: %s\n", use_device, device_prop.name);
  }

  auto tprofiling = Clock::now();

  // Declare target storage and copy A and B
  cudaError_t cuErr;
  site *d_a, *d_c;
  su3_matrix *d_b;
  cuErr = cudaMalloc((void **)&d_a, size_a);
  CUCHECK(cuErr, "Unable to allocate array d_a");
  cuErr = cudaMalloc((void **)&d_b, size_b);
  CUCHECK(cuErr, "Unable to allocate array d_b");
  cuErr = cudaMalloc((void **)&d_c, size_c);
  CUCHECK(cuErr, "Unable to allocate array d_c");
  cudaMemcpy(d_a, a.data(), size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size_b, cudaMemcpyHostToDevice);

  profile->host_to_device_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  const int sitesPerBlock = threadsPerBlock / THREADS_PER_SITE;
  const int blocksPerGrid = (total_sites + sitesPerBlock - 1) / sitesPerBlock;

  if (verbose >= 1) {
    printf("Number of blocks set to %d\n", blocksPerGrid);
    printf("Threads per block set to %d\n", threadsPerBlock);
  }

  // benchmark loop
  auto tstart = Clock::now();
  tprofiling = tstart;

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      cudaDeviceSynchronize();
      tstart = Clock::now();
      tprofiling = tstart;
    }
    k_mat_nn<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
  }
  cudaDeviceSynchronize();
  profile->kernel_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  CUCHECK(cudaGetLastError(), "k_mat_nn kernel Failed");

  // copy data back from device
  tprofiling = Clock::now();
  cudaMemcpy(c.data(), d_c, size_c, cudaMemcpyDeviceToHost);
  profile->device_to_host_time= (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  // Deallocate
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return (ttotal /= 1.0e6);
}
