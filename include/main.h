#ifndef CUDAPLAYGROUND_INCLUDE_MAIN_H_
#define CUDAPLAYGROUND_INCLUDE_MAIN_H_
#include <glog/logging.h>

#include "flags.h"
#include "utils/array_view.h"
#include "utils/event.h"
#include "utils/launcher.h"

void Run() {
  Stream stream;
  thrust::device_vector<int> vec_a;
  thrust::device_vector<int> vec_b;
  thrust::device_vector<int> vec_c;

  for (int i = 0; i < 10; i++) {
    vec_a.push_back(i);
    vec_b.push_back(i);
  }
  vec_c.resize(vec_a.size());

  LaunchKernel(
      stream,
      [] __device__(ArrayView<int> a, ArrayView<int> b, ArrayView<int> c) {
        for (int i = TID_1D; i < a.size(); i += TOTAL_THREADS_1D) {
          c[i] = a[i] + b[i];
        }
      },
      ArrayView<int>(vec_a), ArrayView<int>(vec_b), ArrayView<int>(vec_c));
  stream.Sync();
}

#endif  // CUDAPLAYGROUND_INCLUDE_MAIN_H_
