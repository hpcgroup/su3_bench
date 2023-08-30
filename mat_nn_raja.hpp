#include "RAJA/RAJA.hpp"
#include <chai/ManagedArray.hpp>

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c,
                  size_t total_sites, size_t iterations, size_t threads_per_workgroup, int device) {
    chai::ManagedArray<site>* d_a;
    chai::ManagedArray<su3_matrix>* d_b;
    chai::ManagedArray<site>* d_c;

    return 0;
}
