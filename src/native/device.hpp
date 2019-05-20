#ifndef DEVICE_H
#define DEVICE_H

#include <napi.h>
#include <cuda.h>

class Device : public Napi::ObjectWrap<Device> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Array GetDevices(const Napi::CallbackInfo& info);

  Device(const Napi::CallbackInfo& info);

  int m_device;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value GetName(const Napi::CallbackInfo& info);
  Napi::Value GetTotalMem(const Napi::CallbackInfo& info);
  Napi::Value GetComputeCapability(const Napi::CallbackInfo& info);
};

#endif
