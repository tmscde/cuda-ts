import * as cuda from "..";

let context: cuda.Context;
let mod: cuda.Module;

beforeAll(() => {
  // Use the first available device (might fail depending on environment)
  const device = cuda.getDevices()[0];

  console.log("Running tests using device: ", {
    name: device.getName(),
    totalMem: device.getTotalMem(),
    computeCapability: device.getComputeCapability(),
  });

  context = cuda.createContext(device);

  // Load the module used in subsequent tests
  mod = context.loadModule("dist/__tests__/test.cubin");
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

it("should add two buffers", () => {
  const float32Len = 4;
  const valueCount = 2;
  const bufLen = float32Len * valueCount;

  const buf1 = new Float32Array([12, 1032]);
  const buf2 = new Float32Array([3, 5]);

  // We allocate the first buffer ourselves (this should not be deallocated implicitly)
  const gpuBuf1 = context.allocMem(bufLen);
  gpuBuf1.copyHostToDevice(buf1.buffer);

  const output = context.allocMem(bufLen);
  const func = mod.getFunction("add");

  context.launchKernel(func, [gpuBuf1, buf2], [output]);

  const outBuffer = new ArrayBuffer(bufLen);
  output.copyDeviceToHost(outBuffer);

  // Free the allocated gpu buffer
  gpuBuf1.free();
  output.free();

  const view = new DataView(outBuffer);
  expect(view.getFloat32(0, true)).toBe(15);
  expect(view.getFloat32(4, true)).toBe(1037);
});
