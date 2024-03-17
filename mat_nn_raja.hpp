#include "RAJA/RAJA.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#define THREADS_PER_SITE 36

#if defined(RAJA_ENABLE_CUDA)
  using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
  using teams_x = RAJA::LoopPolicy<RAJA::cuda_block_x_loop>;
  using threads_x = RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;
  using threads_y = RAJA::LoopPolicy<RAJA::cuda_thread_y_direct>;
  using threads_z = RAJA::LoopPolicy<RAJA::cuda_thread_z_direct>;
#elif defined(RAJA_ENABLE_HIP)
  using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<false>>;
  using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_loop>;
  using threads_x = RAJA::LoopPolicy<RAJA::hip_thread_x_direct>;
  using threads_y = RAJA::LoopPolicy<RAJA::hip_thread_y_direct>;
  using threads_z = RAJA::LoopPolicy<RAJA::hip_thread_z_direct>;
#endif

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, size_t total_sites, size_t iterations, size_t threadsPerBlock, int device, Profile* profile) {
  size_t size_a = sizeof(site) * total_sites;
  size_t size_b = sizeof(su3_matrix) * 4;
  size_t size_c = sizeof(site) * total_sites;

  auto tprofiling = Clock::now();

  auto &rm = umpire::ResourceManager::getInstance();
  auto host_alloc = rm.getAllocator("HOST");
  auto strategy = host_alloc.getAllocationStrategy();
  auto device_alloc = rm.getAllocator("DEVICE");

  if (threadsPerBlock == 0)
    threadsPerBlock = THREADS_PER_SITE;

  umpire::util::AllocationRecord record_a{a.data(), size_a, strategy}; 
  umpire::util::AllocationRecord record_b{b.data(), size_b, strategy}; 
  umpire::util::AllocationRecord record_c{c.data(), size_c, strategy}; 

  rm.registerAllocation(a.data(), record_a);
  rm.registerAllocation(b.data(), record_b);
  rm.registerAllocation(c.data(), record_c);

  site *d_a = static_cast<site*>(device_alloc.allocate(size_a));
  su3_matrix *d_b = static_cast<su3_matrix*>(device_alloc.allocate(size_b));
  site *d_c = static_cast<site*>(device_alloc.allocate(size_c));

  rm.copy(d_a, a.data());
  rm.copy(d_b, b.data());
  rm.copy(d_c, c.data());

  profile->host_to_device_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  const int sites_per_block = threadsPerBlock / THREADS_PER_SITE;
  const int teams = RAJA_DIVIDE_CEILING_INT(total_sites, sites_per_block);

  auto tstart = Clock::now();
  tprofiling = tstart;
  
  for (size_t iters = 0; iters < iterations + warmups; ++iters) {
    if (iters == warmups) {
      tstart = Clock::now();
      tprofiling = tstart;
    }
    RAJA::launch<launch_policy>(RAJA::ExecPlace::DEVICE,
      RAJA::LaunchParams(RAJA::Teams(teams), RAJA::Threads(sites_per_block * 4,3,3)),
        [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {
          RAJA::loop<teams_x>(ctx, RAJA::TypedRangeSegment<int>(0, (teams)), [&] (int site) {
            RAJA::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0, sites_per_block * 4), [&] (int j) {
              RAJA::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0, 3), [&] (int k) {
                RAJA::loop<threads_z>(ctx, RAJA::TypedRangeSegment<int>(0, 3), [&] (int l) {
                  const int site_id = j / sites_per_block;
                  const int my_site = (site * sites_per_block) + site_id;
                  const int jj = j % 4;
                  if ( my_site < total_sites ) {
                    Complx cc = {0.0, 0.0};
                    for (int m = 0; m < 3; m++) {
                      cc += d_a[my_site].link[jj].e[k][m] * d_b[jj].e[m][l];
                    }
                    d_c[my_site].link[jj].e[k][l] = cc;
                  }
                });
              });
           });
        });
      });
  }
  profile->kernel_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  tprofiling = Clock::now();
  rm.copy(c.data(), d_c);
  profile->device_to_host_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  return (ttotal /= 1.0e6);
}
