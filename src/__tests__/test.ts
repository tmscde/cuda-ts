import * as cuda from "..";
import { CUevent_flags } from "../enums";

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
  const data = cuda.compileCuToPtx(`
    __device__ __forceinline__ float addf(float* input1, float* input2) {
      return input1[blockIdx.x] + input2[blockIdx.x];
    }

    extern "C" __global__ void add(float c, float* __restrict__ input1, float* __restrict__ input2, float* __restrict__ output) {
      output[blockIdx.x] = addf(input1, input2) + c;
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

// it("should load modules from disk", () => {
//   context.loadModule("dist/__tests__/test.cubin");
// });

it("should return attributes", () => {
  const smCount = device.getAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  expect(smCount).toBeGreaterThan(0);
});

function add(stream: cuda.Stream | 0, sync: () => void) {
  const float32Len = 4;
  const valueCount = 2;
  const bufLen = float32Len * valueCount;

  // Allocate two buffers that will be added together by the kernel on the GPU
  const buf1 = new Float32Array([12, 1032]);
  const gpuBuf1 = context.allocMem(bufLen);
  const buf2 = new Float32Array([3, 5]);
  const gpuBuf2 = context.allocMem(bufLen);

  // Allocate GPU memory for the result
  const output = context.allocMem(bufLen);

  // Get the function kernel
  const func = mod.getFunction("add");

  // Copy the host buffers to the GPU
  gpuBuf1.copyHostToDevice(buf1.buffer, stream);
  gpuBuf2.copyHostToDevice(buf2.buffer, stream);

  // Launch the kernel on the GPU, buf2 is managed (allocated/deallocated) by cuda-ts
  func.launchKernel(
    [1.2, gpuBuf1, gpuBuf2, output], // The kernel parameters
    { x: valueCount, y: 1, z: 1 }, // Dimensions of grid in blocks
    { x: 1, y: 1, z: 1 }, // Dimensions of each thread block
    stream, // The stream to run the kernel on
  );

  sync();

  // Allocate host space for the result
  const outBuffer = new ArrayBuffer(bufLen);

  // Copy the results GPU buffer to host memory
  output.copyDeviceToHost(outBuffer, stream);

  // Free the allocated gpu buffers
  gpuBuf1.free();
  output.free();

  const view = new DataView(outBuffer);
  expect(Math.round(view.getFloat32(0 * float32Len, true) * 10) * 0.1).toBe(12 + 3 + 1.2);
  expect(Math.round(view.getFloat32(1 * float32Len, true) * 10) * 0.1).toBe(1032 + 5 + 1.2);
}

it("should add two buffers on default stream", () => {
  add(0, () => {});
});

it("should add two buffers on non-blocking stream", () => {
  const stream = cuda.createStream(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING);

  add(stream, () => {
    const event = cuda.createEvent(CUevent_flags.CU_EVENT_DEFAULT);
    event.record(stream);
    event.synchronize();
    event.destroy();
  });

  stream.destroy();
});
