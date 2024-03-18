// OpenMP target offload implementation
#include <omp.h>
#include <unistd.h>

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c,
		  size_t total_sites, size_t iterations, size_t threads_per_team, int use_device, Profile* profile) {
  site *d_a = std::data(a);
  su3_matrix *d_b = std::data(b);
  site *d_c = std::data(c);

  size_t size_a = std::size(a);
  size_t size_b = std::size(b);
  size_t size_c = std::size(c);

  auto tprofiling = Clock::now();

  auto tstart = Clock::now();

  #pragma omp target data \
  map(to: d_a[:size_a], d_b[:size_b]) map(from: d_c[:size_c]) 
  for (size_t iters = 0; iters < iterations + warmups; iters++) {
    if (iters == warmups) {
      tstart = Clock::now();
      tprofiling = tstart;
    }

    #pragma omp target teams loop collapse(4)
    for (size_t i = 0; i < total_sites; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            Complx cc = {0.0, 0.0};
            #pragma omp loop bind(thread)
            for (int m = 0; m < 3; m++) {
              cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
            }
            d_c[i].link[j].e[k][l] = cc;
          }
        }
      }
    }
  }

  profile->kernel_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  return ttotal;
}
