# cuda-ts

NVIDIA CUDA™ bindings exposed in TypeScript. Run kernels on the GPU from TypeScript.

## Prerequisites / Requirements

* [CUDA Drivers and Toolkit](https://developer.nvidia.com/cuda-downloads) installed for your platform
* CUDA compatible GPU

## Installing

npm install cuda-ts

## Usage

#### index.ts

```typescript
import * as cuda from "cuda-ts";

// Get a GPU device of choice
const device = cuda.getDevices()[0];

// Create a context for the device
const context = cuda.createContext(device);

// Load the CUDA module
const mod = context.loadModule("add.cubin");

const float32Len = 4;
const valueCount = 2;
const bufLen = float32Len * valueCount;

// Allocate two buffers that will be added together by the kernel on the GPU
const buf1 = new Float32Array([12, 1032]);
const buf2 = new Float32Array([3, 5]);

// In this example, we'll allocate one GPU buffer ourselves for re-use if needed (we need to free the buffer once we're done)
const gpuBuf1 = context.allocMem(bufLen);

// Copy the contents of buf1 to the allocated GPU memory
gpuBuf1.copyHostToDevice(buf1.buffer);

// Allocate GPU memory for a buffer that will hold the result
const output = context.allocMem(bufLen);

// Get the function kernel
const func = mod.getFunction("add");

// Launch the kernel on the GPU, buf2 is managed (allocated/deallocated) by cuda-ts
context.launchKernel(
  func, // The kernel function
  [gpuBuf1, buf2], // The input buffers
  [output], // The output buffer
  { x: output.length / float32Len, y: 1, z: 1 }, // Dimensions of grid in blocks
  { x: 1, y: 1, z: 1 }, // Dimensions of each thread block
);

// Allocate host space for the result
const outBuffer = new ArrayBuffer(bufLen);

// Copy the results GPU buffer to host memory
output.copyDeviceToHost(outBuffer);

// Free the allocated gpu buffers
gpuBuf1.free();
output.free();

// Unload loaded module
mod.unload();

// Destroy the GPU context
context.destroy();

// Print the result
const view = new DataView(outBuffer);
console.log(`12 + 3 = ${view.getFloat32(0, true)}`);
```

#### add.cu

```c
/* Simple CUDA kernel to add two buffers */
extern "C" __global__ void add(float* __restrict__ input1, float* __restrict__ input2, float* __restrict__ output) {
  output[blockIdx.x] = input1[blockIdx.x] + input2[blockIdx.x];
}
```

compile add.cu using the nvcc compiler

```console
nvcc -cubin -o add.cubin add.cu
```
