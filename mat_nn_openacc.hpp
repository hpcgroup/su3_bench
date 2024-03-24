// OpenACC implementation
double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
		  size_t total_sites, size_t iterations, size_t threads_per_team, int use_device, Profile* profile) {
  site * __restrict__ d_a = std::data(a);
  su3_matrix * __restrict__ d_b = std::data(b);
  site * __restrict__ d_c = std::data(c);

  size_t size_a = std::size(a);
  size_t size_b = std::size(b);
  size_t size_c = std::size(c);

  auto tprofiling = Clock::now();

  auto tstart = Clock::now();

  #pragma acc data copyin(d_a[:size_a], d_b[:size_b]) copyout(d_c[:size_c])
  for (size_t iters = 0; iters < iterations + warmups; iters++) {
    if (iters == warmups) {
      tstart = Clock::now();
      tprofiling = tstart;
    }

    #pragma acc parallel loop collapse(4)
    for (int i = 0; i < total_sites; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            Complx cc{};
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
