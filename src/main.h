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

void work_list_swap(Stream& stream) {
  Queue<int> queue1, queue2;
  int max_elem = 1000;
  queue1.Init(100 * 100);  // capacity
  queue2.Init(100 * 100);  // capacity

  queue1.Clear(stream);
  queue2.Clear(stream);

  std::vector<int> host_data;

  host_data.push_back(1);
  host_data.push_back(2);
  host_data.push_back(3);
  host_data.push_back(4);
  host_data.push_back(5);

  CHECK_CUDA(cudaMemcpyAsync(queue1.data(), host_data.data(),
                             sizeof(int) * host_data.size(),
                             cudaMemcpyHostToDevice, stream.cuda_stream()));
  queue1.set_size(stream, host_data.size());

  for (int i = 0; i < 10; i++) {
    queue2.Clear(stream);

    auto d_in_queue = queue1.DeviceObject();  // should be called after Init
    auto d_out_queue = queue2.DeviceObject();

    LaunchKernel(
        stream,
        [=] __device__(int max) mutable {
          for (int i = TID_1D; i < d_in_queue.size(); i += TOTAL_THREADS_1D) {
            d_out_queue.Append(d_in_queue[i] + 1);
          }
        },
        max_elem);
    queue1.Swap(queue2);
  }

  stream.Sync();
  auto size = queue1.size(stream);
  std::cout << "queue size: " << size << std::endl;

  thrust::host_vector<int> h_data(size);

  CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(h_data.data()),
                             queue1.data(), size * sizeof(int),
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
  work_list_swap(stream);
}

#endif  // CUDAPLAYGROUND_INCLUDE_MAIN_H_
