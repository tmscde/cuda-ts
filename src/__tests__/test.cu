#include <cstdio>

extern "C" {
  __global__ void add(float* input1, float* input2, float* output) {
    #if __CUDA_ARCH__ >= 200
    printf("GPU: Running thread (%d,%d,%d) in blocks (%d,%d,%d).\n",
      threadIdx.x, threadIdx.y, threadIdx.z,
      blockIdx.x, blockIdx.y, blockIdx.z);
    #endif

    output[blockIdx.x] = input1[blockIdx.x] + input2[blockIdx.x];
  }
}
