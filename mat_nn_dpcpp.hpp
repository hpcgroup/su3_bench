// Intel DPCPP implementation
#include <sycl/sycl.hpp>

#define THREADS_PER_SITE 36

// Sycl requires that kernels be named
class k_mat_nn;

double su3_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b, std::vector<site> &c, 
		  const size_t total_sites, const size_t iterations, size_t wgsize, const int target, Profile* profile)
{ 
  // build a list of devices
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
  std::vector<sycl::device> devices;
  for (size_t i=0, d=0; i < platforms.size(); ++i) {
    std::vector<sycl::device> pdevices = platforms[i].get_devices();
    for (size_t j=0; j < pdevices.size(); ++j, ++d) {
      devices.insert(devices.end(), pdevices[j]);
      if (verbose >= 3)
        std::cout << "Appending device " << d << ": " << pdevices[j].get_info<sycl::info::device::name>() << std::endl;
    }
  }

  sycl::queue queue{sycl::device::get_devices()[target]};
  if (verbose >= 2)
    std::cout << "Using device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

  // FYI, look at device maximums
  if (verbose >= 3) {
    std::cout << "max compute units = " 
       << queue.get_device().get_info<sycl::info::device::max_compute_units>() << "\n";
    std::cout << "max workgroup size = " 
       << queue.get_device().get_info<sycl::info::device::max_work_group_size>() << "\n";
  }

  // check to make sure the workgroup size is sufficient for the algorithm
  if (wgsize == 0)
    wgsize = THREADS_PER_SITE;

  // set the total number of work items
  size_t total_wi = total_sites * THREADS_PER_SITE;
  if (verbose >= 3) {
    std::cout << "Setting number of work items " << total_wi << std::endl;
    std::cout << "Workgroup size is " << wgsize << std::endl;
  }

  std::cout << std::flush;

  auto tprofiling = Clock::now();

  // allocate device memory
  site*       d_a = (site*)       malloc_device(total_sites * sizeof(site), queue);
  su3_matrix* d_b = (su3_matrix*) malloc_device(4 * sizeof(su3_matrix), queue);
  site*       d_c = (site*)       malloc_device(total_sites * sizeof(site), queue);
  if (d_a == NULL || d_b == NULL || d_c == NULL) {
    std::cout << "Unable to allocate device memory " << std::endl;
    exit(1);
  }

  // Move host side memory to device allocated buffers
  queue.memcpy(d_a, a.data(), a.size() * sizeof(site));
  queue.memcpy(d_b, b.data(), b.size() * sizeof(su3_matrix));
  queue.wait();

  profile->host_to_device_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  // benchmark loop
  auto tstart = Clock::now();
  tprofiling = tstart;
  for (size_t iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      queue.wait();
      tstart = Clock::now();
      tprofiling = tstart;
    }

    // create a command_group to issue commands
    queue.submit([&](sycl::handler& cgh) {
      // Lambda function defines the kernel scope
      cgh.parallel_for<class k_mat_nn>(
      sycl::nd_range<1> {total_wi, wgsize}, [=](sycl::nd_item<1> item) {
        size_t myThread = item.get_global_id(0);
        size_t mySite = myThread/36;
        if (mySite < total_sites) {
          int j = (myThread%36)/9;
          int k = (myThread%9)/3;
          int l = myThread%3;
          Complx cc = {0.0, 0.0};
          for (int m=0;m<3;m++) {
            const auto aa = d_a[mySite].link[j].e[k][m];
            const auto bb = d_b[j].e[m][l];
#ifndef MILC_COMPLEX
            cc += aa * bb;
#else
            CMULSUM(aa, bb, cc);
#endif
          }
          d_c[mySite].link[j].e[k][l] = cc;
        }
      }); // end of the kernel lambda function
    });   // end of command group
  queue.wait();
  } // end of iteration loop

  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  profile->kernel_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  // Move the result back to the host side vector
  tprofiling = Clock::now();
  queue.memcpy(c.data(), d_c, c.size() * sizeof(site));
  queue.wait();
  profile->device_to_host_time= (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  free(d_a, queue);
  free(d_b, queue);
  free(d_c, queue);

  return (ttotal /= 1.0e6);
} // end of SYCL block
