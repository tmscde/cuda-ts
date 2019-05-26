#include "device.hpp"
#include "shared.hpp"

Napi::FunctionReference Device::constructor;

Napi::Object Device::Init(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "Device", {
                                                       InstanceMethod("getName", &Device::GetName),
                                                       InstanceMethod("getTotalMem", &Device::GetTotalMem),
                                                       InstanceMethod("getComputeCapability", &Device::GetComputeCapability),
                                                       InstanceMethod("getAttribute", &Device::GetAttribute),
                                                   });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("Device", func);
  exports.Set("getDevices", Napi::Function::New(env, GetDevices));

  return exports;
}

Napi::Array Device::GetDevices(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  int count = 0;
  if (!validate(cuDeviceGetCount(&count), env))
  {
    return Napi::Array::New(info.Env(), count);
  }

  Napi::Array devices = Napi::Array::New(info.Env(), count);
  for (int i = 0; i < count; i++)
  {
    Napi::Object obj = constructor.New({});
    Napi::ObjectWrap<Device>::Unwrap(obj)->m_device = i;
    devices[i] = obj;
  }

  return devices;
}

Device::Device(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Device>(info)
{
}

Napi::Value Device::GetName(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  char deviceName[256];

  if (!validate(cuDeviceGetName(deviceName, 256, m_device), env))
  {
    return env.Undefined();
  }

  return Napi::String::New(env, deviceName);
}

Napi::Value Device::GetTotalMem(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  size_t totalGlobalMem;

  if (!validate(cuDeviceTotalMem(&totalGlobalMem, m_device), env))
  {
    return env.Undefined();
  }

  return Napi::Number::New(env, totalGlobalMem);
}

Napi::Value Device::GetComputeCapability(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  int major = 0, minor = 0;

  if (!validate(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, m_device), env))
  {
    return env.Undefined();
  }

  if (!validate(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, m_device), env))
  {
    return env.Undefined();
  }

  char version[256];
  sprintf(version, "%d.%d", major, minor);
  return Napi::String::New(env, version);
}

Napi::Value Device::GetAttribute(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();
  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Number arg0 = info[0].As<Napi::Number>();

  int value = 0;
  if (!validate(cuDeviceGetAttribute(&value, (CUdevice_attribute)arg0.Int32Value(), m_device), env))
  {
    return env.Undefined();
  }

  return Napi::Number::New(env, value);
}
