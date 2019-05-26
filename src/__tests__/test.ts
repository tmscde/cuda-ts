import * as cuda from "..";

let context: cuda.Context;
let mod: cuda.Module;
let device: cuda.Device;

beforeAll(() => {
  // Use the first available device (might fail depending on environment)
  device = cuda.getDevices()[0];

  console.log("Running tests using device: ", {
    name: device.getName(),
    totalMem: device.getTotalMem(),
    computeCapability: device.getComputeCapability(),
  });

  context = cuda.createContext(device);

  // Load the module used in subsequent tests
  // mod = context.loadModule("dist/__tests__/test.cubin");
  const data = cuda.compileCuToPtx(`
    __device__ __forceinline__ float add(float* input1, float* input2) {
      return input1[blockIdx.x] + input2[blockIdx.x];
    }

    extern "C" __global__ void add(float* __restrict__ input1, float* __restrict__ input2, float* __restrict__ output) {
      output[blockIdx.x] = add(input1, input2);
    }
  `);

  mod = context.loadModuleData(data);
});

afterAll(() => {
  if (mod) {
    mod.unload();
  }
  if (context) {
    context.destroy();
  }
});

it("should compile cu to ptx", () => {
  cuda.compileCuToPtx(`
    extern "C" __global__ void add(float* __restrict__ input1, float* __restrict__ input2, float* __restrict__ output) {
      output[blockIdx.x] = input1[blockIdx.x] + input2[blockIdx.x];
    }
  `);
});

it("should return attributes", () => {
  const smCount = device.getAttribute(cuda.CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  expect(smCount).toBeGreaterThan(0);
});

it("should add two buffers", () => {
  const float32Len = 4;
  const valueCount = 2;
  const bufLen = float32Len * valueCount;

  // Allocate two buffers that will be added together by the kernel on the GPU
  const buf1 = new Float32Array([12, 1032]);
  const buf2 = new Float32Array([3, 5]);

  // We allocate the first buffer ourselves (we need to free the buffer once we're done)
  const gpuBuf1 = context.allocMem(bufLen);

  // Copy the buffer contents to GPU memory
  gpuBuf1.copyHostToDevice(buf1.buffer);

  // Allocate GPU memory for the result
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

  const view = new DataView(outBuffer);
  expect(view.getFloat32(0, true)).toBe(15);
  expect(view.getFloat32(4, true)).toBe(1037);
});
