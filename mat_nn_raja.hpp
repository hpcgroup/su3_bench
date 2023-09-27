#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
#include <chai/ManagedArray.hpp>

#if defined(RAJA_ENABLE_OPENMP)
using policy = RAJA::omp_parallel_for_exec;
#elif defined(RAJA_ENABLE_CUDA)
  using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
  using teams_x = RAJA::LoopPolicy<RAJA::cuda_block_x_loop>;
  using threads_x= RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;
  using threads_y= RAJA::LoopPolicy<RAJA::cuda_thread_y_direct>;
  using threads_z= RAJA::LoopPolicy<RAJA::cuda_thread_z_direct>;
#elif defined(RAJA_ENABLE_HIP)
  using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<false>>;
  using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_loop>;
  using threads_x= RAJA::LoopPolicy<RAJA::hip_thread_x_direct>;
  using threads_y= RAJA::LoopPolicy<RAJA::hip_thread_y_direct>;
  using threads_z= RAJA::LoopPolicy<RAJA::hip_thread_z_direct>;
#endif


double su3_mat_nn(chai::ManagedArray<site>& a, chai::ManagedArray<su3_matrix>& b, chai::ManagedArray<site> &c,
    size_t total_sites, size_t iterations, size_t threads_per_workgroup, int device) {
  auto timer = RAJA::Timer();
  for (size_t iters = 0; iters < iterations + warmups; ++iters) {

    if (iters == warmups) {
      timer.start();
    }
  RAJA::launch<launch_policy>(RAJA::ExecPlace::DEVICE,
      RAJA::LaunchParams(RAJA::Teams(total_sites), RAJA::Threads(4,3,3)),
      [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {
        RAJA::loop<teams_x>(ctx, RAJA::TypedRangeSegment<int>(0, total_sites), [&] (int site) {
           RAJA::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0, 4), [&] (int j) {
             RAJA::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0, 3), [&] (int k) {
                RAJA::loop<threads_z>(ctx, RAJA::TypedRangeSegment<int>(0, 3), [&] (int l) {
                  Complx cc = {0.0, 0.0};
                  for (int m = 0; m < 3; m++) {
                    cc += a[site].link[j].e[k][m] * b[j].e[m][l];
                  }
                  c[site].link[j].e[k][l] = cc;
           });
         });
       });
     });
     });
  }
  timer.stop();
  RAJA::Timer::ElapsedType elapsed = timer.elapsed();
  c.move(chai::CPU);

  return elapsed;
}
