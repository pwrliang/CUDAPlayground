```shell
mkdir build
cd build
cmake -DCMAKE_CUDA_COMPILER=$(which nvcc) ..
make
./main
```