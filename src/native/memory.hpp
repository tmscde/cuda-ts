#ifndef MEMORY_H
#define MEMORY_H

#include <napi.h>
#include <cuda.h>

class Memory : public Napi::ObjectWrap<Memory> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New();

  Memory(const Napi::CallbackInfo& info);

  CUdeviceptr m_ptr;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value CopyHostToDevice(const Napi::CallbackInfo& info);
  Napi::Value CopyDeviceToHost(const Napi::CallbackInfo& info);
  Napi::Value Free(const Napi::CallbackInfo& info);
};

#endif
