#include <cstdio>

__device__ __forceinline__ float addf(float* input1, float* input2) {
  return input1[blockIdx.x] + input2[blockIdx.x];
}

extern "C" __global__ void add(float c, float* __restrict__ input1, float* __restrict__ input2, float* __restrict__ output) {
  output[blockIdx.x] = addf(input1, input2) + c;
}
