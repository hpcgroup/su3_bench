#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
#include <chai/ManagedArray.hpp>

#if defined(RAJA_ENABLE_OPENMP)
using policy = RAJA::omp_parallel_for_exec;
#elif defined(RAJA_ENABLE_CUDA)
using policy = RAJA::cuda_exec<256>;
#elif defined(RAJA_ENABLE_HIP)
using policy = RAJA::hip_exec<256>;
#endif


double su3_mat_nn(chai::ManagedArray<site>& a, chai::ManagedArray<su3_matrix>& b, chai::ManagedArray<site> &c,
                  size_t total_sites, size_t iterations, size_t threads_per_workgroup, int device) {
    RAJA::RangeSegment range(0, total_sites);

    auto timer = RAJA::Timer();
    for (int iters = 0; iters < iterations + warmups; iters++) {
        if (iters == warmups) {
            timer.start();
        }

        RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE (int i) {
            int j = (i % 36) / 9;
            int k = (i % 9) / 3;
            int l = i % 3;

            Complx cc = {0.0, 0.0};
            for (int m = 0; m < 3; m++) {
                cc += a[i].link[j].e[k][m] * b[j].e[m][l];
            }
            c[i].link[j].e[k][l] = cc;
        });
    }
    timer.stop();

    RAJA::Timer::ElapsedType elapsed = timer.elapsed();

    return elapsed;
}
