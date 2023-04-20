#ifndef CUDAPLAYGROUND_INCLUDE_MAIN_H_
#define CUDAPLAYGROUND_INCLUDE_MAIN_H_
#include "utils/array_view.h"
#include "utils/event.h"
#include "utils/launcher.h"
#include "utils/queue.h"

void vec_add(Stream& stream) {
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

void work_list(Stream& stream) {
  Queue<int> queue;
  int max_elem = 1000;
  queue.Init(100 * 100);  // capacity

  queue.Clear(stream);

  auto d_queue = queue.DeviceObject();  // should be called after Init

  LaunchKernel(
      stream,
      [=] __device__(int max) mutable {
        for (int i = TID_1D; i < max; i += TOTAL_THREADS_1D) {
          d_queue.Append(i);
        }
      },
      max_elem);
  stream.Sync();
  auto size = queue.size(stream);
  std::cout << "queue size: " << size << std::endl;

  thrust::host_vector<int> h_data(size);

  CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(h_data.data()),
                             queue.data(), size * sizeof(int),
                             cudaMemcpyDeviceToHost, stream.cuda_stream()));
  stream.Sync();

  for (int i = 0; i < size; i++) {
    std::cout << "idx: " << i << " data: " << h_data[i] << std::endl;
  }
}

void Run() {
  Stream stream;

  vec_add(stream);
  work_list(stream);
}

#endif  // CUDAPLAYGROUND_INCLUDE_MAIN_H_
