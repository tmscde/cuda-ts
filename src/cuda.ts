const cuda = require("bindings")("cuda");

export interface Device {
  getName(): string;
  getTotalMem(): number;
  getComputeCapability(): string;
}

export interface Function {
  launchKernel(buffers: GpuBuffer[]): void;
}

export interface Module {
  getFunction(name: string): Function;
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
  loadModuleFromCu(cu: string): Module;
  launchKernel(func: Function, input: (Float32Array | GpuBuffer)[], output: (number | GpuBuffer)[]): Float32Array[];
  destroy(): void;
}

class FunctionImpl implements Function {
  constructor(private func: any) {}

  launchKernel(buffers: GpuBuffer[]): void {
    const gpuMem = buffers.map(x => (x as GpuBufferImpl).mem);
    return this.func.launchKernel(gpuMem);
  }
}

class ModuleImpl {
  constructor(private mod: any) {}

  getFunction(name: string): Function {
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

  loadModuleFromCu(filename: string) {
    return new ModuleImpl(this.context.moduleLoad(filename));
  }

  launchKernel(func: Function, input: (Float32Array | GpuBuffer)[], output: (number | GpuBuffer)[]): Float32Array[] {
    const buffersToFree: GpuBuffer[] = [];
    const inputBuffers = input.map(inputBuffer => {
      if (!isFloat32Array(inputBuffer)) {
        return inputBuffer;
      }

      const gpuBuffer = this.allocMem(inputBuffer.byteLength);
      gpuBuffer.copyHostToDevice(inputBuffer.buffer);
      buffersToFree.push(gpuBuffer);
      return gpuBuffer;
    });

    const outputBuffers = output.map(buffer => {
      if (!isNumber(buffer)) {
        return buffer;
      }

      const gpuBuffer = this.allocMem(buffer);
      buffersToFree.push(gpuBuffer);
      return gpuBuffer;
    });

    func.launchKernel([...inputBuffers, ...outputBuffers]);

    // Copy output buffers
    const results = outputBuffers.map(buffer => {
      const hostBuffer = new ArrayBuffer(buffer.length);
      buffer.copyDeviceToHost(hostBuffer);
      return new Float32Array(hostBuffer);
    });

    while (buffersToFree.length > 0) {
      buffersToFree.pop()!.free();
    }

    return results;
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

export function compileCuToPtx(cu: string) {
  return cuda.compileCuToPtx(cu);
}
