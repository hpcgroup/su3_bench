// OpenACC implementation

typedef struct{
	double d2h_time;
	double kernel_time;
	double h2d_time;
} Profile;

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
		  size_t total_sites, size_t iterations, size_t threads_per_team, int use_device, Profile* profile)
{
  site *d_a, *d_c;
  su3_matrix *d_b;
  size_t len_a, len_b, len_c;
  d_a = a.data(); len_a = a.size();
  d_b = b.data(); len_b = b.size();
  d_c = c.data(); len_c = c.size();

  auto tstart = Clock::now();
  auto tprofiling = tstart;

  // Move A, B and C vectors to the device
  #pragma acc enter data copyin(d_a[0:len_a], d_b[0:len_b], d_c[0:len_c])

  profile->h2d_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  // benchmark loop
  tprofiling = Clock::now();

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      tstart = Clock::now();
      tprofiling = Clock::now();
    }

    #pragma acc parallel loop gang present(d_a[0:len_a], d_b[0:len_b], d_c[0:len_c])
    for(int i=0;i<total_sites;++i) {
      #pragma acc loop worker vector collapse(3)
      for (int j=0; j<4; ++j) {
        for(int k=0;k<3;k++) {
          for(int l=0;l<3;l++){
	    Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
            #pragma acc loop seq
            for(int m=0;m<3;m++) {
               cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
            }
            d_c[i].link[j].e[k][l] = cc;
#else
            #pragma acc loop seq
            for(int m=0;m<3;m++) {
               CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
            }
            d_c[i].link[j].e[k][l].real = cc.real;
            d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
          }
        }
      }
    }
  }

  profile->kernel_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;
  tprofiling = Clock::now();

  // move the result back 
  #pragma acc exit data copyout(d_c[0:len_c])

  profile->d2h_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  return (ttotal /= 1.0e6);
}
