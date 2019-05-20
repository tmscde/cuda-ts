"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const cuda = require("../src/cuda");
let context;
let mod;
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
    mod = context.loadModule("__tests__/data/test.cubin");
});
afterAll(() => {
    if (mod) {
        mod.unload();
    }
    if (context) {
        context.destroy();
    }
});
it("should add two buffers", () => {
    try {
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
        console.log("Launch kernel");
        context.launchKernel(func, [gpuBuf1, buf2], [output]);
        // const outBuffer = new ArrayBuffer(bufLen);
        // output.copyDeviceToHost(outBuffer);
        // // Free the allocated gpu buffer
        // gpuBuf1.free();
        // output.free();
        // const view = new DataView(outBuffer);
        // expect(view.getFloat32(0)).toBe(15);
        // expect(view.getFloat32(1)).toBe(1037);
    }
    catch (error) {
        console.log("AN ERROR OCCURED", error);
    }
    finally {
        if (mod) {
            mod.unload();
        }
        if (context) {
            context.destroy();
        }
    }
});
//# sourceMappingURL=test.js.map