// Intel DPCPP implementation
#include <CL/sycl.hpp>

#define THREADS_PER_SITE 36

// Sycl requires that kernels be named
class k_mat_nn;

double su3_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b, std::vector<site> &c, 
              const size_t total_sites, const size_t iterations, size_t wgsize, const int target)
{ 
  using namespace cl::sycl;

  // build a list of devices
  std::vector<platform> platforms = platform::get_platforms();
  std::vector<device> devices;
  for (int i=0, d=0; i < platforms.size(); ++i) {
    std::vector<device> pdevices = platforms[i].get_devices();
    for (int j=0; j < pdevices.size(); ++j, ++d) {
      devices.insert(devices.end(), pdevices[j]);
      if (verbose >= 3)
        std::cout << "Appending device " << d << ": " << pdevices[j].get_info<info::device::name>() << std::endl;
    }
  }

  // Create a SYCL queue and set the device
  device target_device;
  if (target < 0) {
    default_selector selector;
    target_device = selector.select_device();
  } 
  else if (target < devices.size()) {
    target_device = devices[target];
  }
  else {
    std::cout << "Invalid device specified: " << target << std::endl;
    exit(1);
  }
  queue queue(target_device);
  if (verbose >= 2)
    std::cout << "Using device: " << queue.get_device().get_info<info::device::name>() << "\n";

  // FYI, look at device maximums
  if (verbose >= 3) {
    std::cout << "max compute units = " 
       << queue.get_device().get_info<info::device::max_compute_units>() << "\n";
    std::cout << "max workgroup size = " 
       << queue.get_device().get_info<info::device::max_work_group_size>() << "\n";
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

#ifdef ALIGNED_WORK
    auto tstart = Clock::now();
#endif

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

  // benchmark loop
#ifndef ALIGNED_WORK
  auto tstart = Clock::now();
#endif
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      queue.wait();
      tstart = Clock::now();
	  }

    // create a command_group to issue commands
    queue.submit([&](handler& cgh) {
      // Lambda function defines the kernel scope
      cgh.parallel_for<class k_mat_nn>(
      nd_range<1> {total_wi, wgsize}, [=](nd_item<1> item) {
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

  // Move the result back to the host side vector
  queue.memcpy(c.data(), d_c, c.size() * sizeof(site));
  queue.wait();

  free(d_a, queue);
  free(d_b, queue);
  free(d_c, queue);

  return (ttotal /= 1.0e6);
} // end of SYCL block

