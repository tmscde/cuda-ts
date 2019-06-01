export * from "./enums";
import { CUdevice_attribute_enum } from "./enums";

const cuda = require("bindings")("cuda");

export interface Dimensions {
  x: number;
  y: number;
  z: number;
}

export interface Device {
  getName(): string;
  getTotalMem(): number;
  getComputeCapability(): string;
  getAttribute(attribute: CUdevice_attribute_enum): number;
}

export interface KernelFunc {
  launchKernel(buffers: GpuBuffer[], gridDim: Dimensions, blockDim: Dimensions): void;
}

export interface Module {
  getFunction(name: string): KernelFunc;
  unload(): void;
}

export interface GpuBuffer {
  readonly length: number;
  copyDeviceToHost(buffer: ArrayBuffer): void;
  copyHostToDevice(buffer: ArrayBuffer): void;
  free(): void;
}

export interface Context {
  allocMem(byteLength: number): GpuBuffer;
  loadModule(filename: string): Module;
  loadModuleData(data: ArrayBuffer): Module;
  /**
   * Invokes the kernel {@param func} on a {@param gridDim.x} x {@param gridDim.y} x {@param gridDim.z} grid of blocks. Each block contains {@param blockDim.x} x {@param blockDim.y} x {@param blockDim.z} threads.
   * @param {KernelFunc} func The kernel function
   * @param {(number | Float32Array | GpuBuffer)[]} input Buffers to be read by the kernel function
   * @param {(number | GpuBuffer)[]} Buffers to be written to by the kernel function
   * @param {Dimensions} gridDim Size of the grid in blocks
   * @param {Dimensions} blockDim Dimensions of each thread block
   * @return {Float32Array[]} The contents of the {@param input} buffers in host memory
   */
  launchKernel(func: KernelFunc, data: (Float32Array | GpuBuffer)[], gridDim: Dimensions, blockDim: Dimensions): void;
  destroy(): void;
}

class FunctionImpl implements KernelFunc {
  constructor(private func: any) {}

  launchKernel(buffers: GpuBuffer[], gridDim: Dimensions, blockDim: Dimensions): void {
    const gpuMem = buffers.map(x => (x as GpuBufferImpl).mem);
    return this.func.launchKernel(gpuMem, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  }
}

class ModuleImpl {
  constructor(private mod: any) {}

  getFunction(name: string): KernelFunc {
    return new FunctionImpl(this.mod.getFunction(name));
  }

  unload(): void {
    this.mod.unload();
  }
}

class GpuBufferImpl implements GpuBuffer {
  constructor(public mem: any, public readonly length: number) {}
  copyDeviceToHost(buffer: ArrayBuffer): void {
    this.mem.copyDeviceToHost(buffer);
  }
  copyHostToDevice(buffer: ArrayBuffer): void {
    this.mem.copyHostToDevice(buffer);
  }
  free() {
    if (this.mem !== null) {
      this.mem.free();
      this.mem = null;
    }
  }
}

function isNumber(val: any): val is number {
  return typeof val === "number";
}

function isFloat32Array(val: any): val is Float32Array {
  return val.constructor === Float32Array;
}

class ContextImpl implements Context {
  constructor(private device: Device, private context: any = cuda.createContext(device)) {}

  allocMem(byteLength: number): GpuBuffer {
    return new GpuBufferImpl(this.context.allocMem(byteLength), byteLength);
  }

  loadModule(filename: string) {
    return new ModuleImpl(this.context.moduleLoad(filename));
  }

  loadModuleData(data: ArrayBuffer) {
    return new ModuleImpl(this.context.moduleLoadData(data));
  }

  launchKernel(func: KernelFunc, data: (Float32Array | GpuBuffer)[], gridDim: Dimensions, blockDim: Dimensions): void {
    const buffersToFree: GpuBuffer[] = [];
    const buffers = data.map(buffer => {
      if (!isFloat32Array(buffer)) {
        return buffer;
      }
      const gpuBuffer = this.allocMem(buffer.byteLength);
      gpuBuffer.copyHostToDevice(buffer.buffer);
      buffersToFree.push(gpuBuffer);
      return gpuBuffer;
    });

    func.launchKernel(buffers, gridDim, blockDim);

    while (buffersToFree.length > 0) {
      buffersToFree.pop()!.free();
    }
  }

  destroy() {
    if (this.context !== null) {
      this.context.destroy();
      this.context = null;
    }
  }
}

export function getDevices(): Device[] {
  return cuda.getDevices();
}

export function createContext(device: Device): Context {
  return new ContextImpl(device);
}

export function compileCuToPtx(cu: string): ArrayBuffer {
  return cuda.compileCuToPtx(cu);
}
