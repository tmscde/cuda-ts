export * from "./enums";
import { CUdevice_attribute, CUstream_flags, CUevent_flags } from "./enums";

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
  getAttribute(attribute: CUdevice_attribute): number;
}

export interface KernelFunc {
  /**
   * Invokes the kernel {@param func} on a {@param gridDim.x} x {@param gridDim.y} x {@param gridDim.z} grid of blocks. Each block contains {@param blockDim.x} x {@param blockDim.y} x {@param blockDim.z} threads.
   * @param {KernelFunc} func The kernel function
   * @param {GpuBuffer[]} buffers Buffers to be used by the kernel function
   * @param {Dimensions} gridDim Size of the grid in blocks
   * @param {Dimensions} blockDim Dimensions of each thread block
   * @param {Stream | 0} stream Stream to perform the launch in, 0 is default stream which is synchronous
   */

  launchKernel(params: (GpuBuffer | number)[], gridDim: Dimensions, blockDim: Dimensions, stream: Stream | 0): void;
}

export interface Module {
  getFunction(name: string): KernelFunc;
  unload(): void;
}

export interface GpuBuffer {
  copyDeviceToHost(buffer: ArrayBuffer, stream: Stream | 0): void;
  copyHostToDevice(buffer: ArrayBuffer, stream: Stream | 0): void;
  free(): void;
}

export interface Context {
  readonly device: Device;

  allocMem(byteLength: number): GpuBuffer;
  loadModule(filename: string): Module;
  loadModuleData(data: ArrayBuffer): Module;
  destroy(): void;
}

export interface Event {
  record(stream: Stream): void;
  synchronize(): void;
  destroy(): void;
}

export interface Stream {
  waitForEvent(event: Event): void;
  destroy(): void;
}

class FunctionImpl implements KernelFunc {
  constructor(private func: any) {}

  launchKernel(params: (GpuBuffer | number)[], gridDim: Dimensions, blockDim: Dimensions, stream: Stream | 0): void {
    return this.func.launchKernel(params, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, stream);
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

class ContextImpl implements Context {
  constructor(public device: Device, private context: any = cuda.createContext(device)) {}

  allocMem(byteLength: number): GpuBuffer {
    return this.context.allocMem(byteLength);
  }

  loadModule(filename: string) {
    return new ModuleImpl(this.context.moduleLoad(filename));
  }

  loadModuleData(data: ArrayBuffer) {
    return new ModuleImpl(this.context.moduleLoadData(data));
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

export function createStream(flags: CUstream_flags): Stream {
  return cuda.createStream(flags);
}

export function createEvent(flags: CUevent_flags): Event {
  return cuda.createEvent(flags);
}

export function compileCuToPtx(cu: string, options?: string[]): ArrayBuffer {
  return cuda.compileCuToPtx(cu, options);
}

export function getSizeofCurandState() {
  return cuda.getSizeofCurandState();
}
