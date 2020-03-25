// OpenMP target offload implementation
#include <omp.h>
#include <unistd.h>
#define USE_WORKAROUND 2
#define THREADS_PER_SITE 36
#define DEFAULT_NUM_TEAMS 1600

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
              size_t total_sites, size_t iterations, size_t threads_per_team, int use_device)
{
  if (threads_per_team < THREADS_PER_SITE)
    threads_per_team = THREADS_PER_SITE;

  site *d_a, *d_c;
  su3_matrix *d_b;
  size_t len_a, len_b, len_c;
  d_a = a.data(); len_a = a.size();
  d_b = b.data(); len_b = b.size();
  d_c = c.data(); len_c = c.size();
 
  // Move A and B data to the device, Allocate C data
  double ttotal;
  #pragma omp target data map(to: d_a[0:len_a], d_b[0:len_b]) map(from: d_c[0:len_c])
  {  // begin OpenMP block

  // benchmark loop
  auto tstart = Clock::now();
#if USE_WORKAROUND == 1
  // This version improves performance over the baseline
  // Contributed by Chris Daley, NERSC
  size_t num_teams = DEFAULT_NUM_TEAMS;
  int opt;
  optind = 1;
  while ((opt=getopt(g_argc, g_argv, ":n:")) != -1) {
    switch (opt) {
    case 'n':
      num_teams = atoi(optarg);
      break;
    }
  }

  if (verbose >= 1) {
    std::cout << "Number of teams = " << num_teams << std::endl;
    std::cout << "Threads per team = " << threads_per_team << std::endl;
  }

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();
    #pragma omp target teams num_teams(num_teams) thread_limit(threads_per_team)
    {
      #pragma omp parallel
      {
        int total_teams = omp_get_num_teams();
        int team_id = omp_get_team_num();
        int sites_per_team = (total_sites + total_teams - 1) / total_teams;
        int istart = team_id * sites_per_team;
        if (istart > total_sites) istart = total_sites;
        int iend = istart + sites_per_team;
        if (iend > total_sites) iend = total_sites;

        for (int i = istart; i < iend; ++i) {
          #pragma omp for collapse(3)
          for (int j=0; j<4; ++j) {
            for(int k=0;k<3;k++) {
              for(int l=0;l<3;l++){
                Complx cc = {0.0, 0.0};
#ifndef LAT_CHECK
#ifndef MILC_COMPLEX
                for(int m=0;m<3;m++) {
                  cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
                }
                d_c[i].link[j].e[k][l] = cc;
#else
                for(int m=0;m<3;m++) {
                   CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
                }
                d_c[i].link[j].e[k][l].real = cc.real;
                d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
#endif  // LAT_CHECK
              }
            }
          }
        }  // end of i loop
      }  // end of parallel region
    }  // end of teams region
  }

#elif USE_WORKAROUND == 2
  // This code improves performance over above baseline
  // Similar to Cuda and OpenCL work item approach
  // Initial contribution by Xinmin Tian, Intel
  size_t num_work_items = total_sites * THREADS_PER_SITE;
  size_t num_teams = num_work_items / (double)threads_per_team + 0.999999;

  if (verbose >= 1) {
    std::cout << "Number of teams = " << num_teams << std::endl;
    std::cout << "Threads per team = " << threads_per_team << std::endl;
    std::cout << "Number of work items = " << num_work_items << std::endl;
  }

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();
    #pragma omp target teams distribute parallel for num_teams(num_teams)  thread_limit(threads_per_team)
    for (int id =0; id < num_work_items; id++) {
      int i = id/36;
      if (i < total_sites) {
        int j = (id%36)/9;
        int k = (id%9)/3;
        int l = id%3;

        Complx cc = {0.0, 0.0};
#ifndef LAT_CHECK
#ifndef MILC_COMPLEX
        for(int m=0;m<3;m++) {
          cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
        }
        d_c[i].link[j].e[k][l] = cc;
#else
        for(int m=0;m<3;m++) {
           CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
        }
        d_c[i].link[j].e[k][l].real = cc.real;
        d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
#endif  // LAT_CHECK
      }
    }
  }

#else
  // Baseline implementation
  // Uses the purest intent of OpenMP, but has performance issues
  // See USE_WORLAROUND
  size_t num_teams = total_sites * THREADS_PER_SITE / (double)threads_per_team + 0.999999;

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();
    #pragma omp target teams distribute num_teams(num_teams) thread_limit(threads_per_team)
    for(int i=0;i<total_sites;++i) {
      #pragma omp parallel for collapse(3)
      for (int j=0; j<4; ++j) {
        for(int k=0;k<3;k++) {
          for(int l=0;l<3;l++){
            Complx cc = {0.0, 0.0};
#ifndef LAT_CHECK
#ifndef MILC_COMPLEX
            for(int m=0;m<3;m++) {
               cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
            }
            d_c[i].link[j].e[k][l] = cc;
#else
            for(int m=0;m<3;m++) {
               CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
            }
            d_c[i].link[j].e[k][l].real = cc.real;
            d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
#endif  // LAT_CHECK
          }
        }
      }
    }
  }
#endif

  ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  } // end of OpenMP block, C gets moved back to the host

  return (ttotal /= 1.0e6);
}
