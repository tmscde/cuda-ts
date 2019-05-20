#include "memory.hpp"
#include "shared.hpp"

Napi::FunctionReference Memory::constructor;

Napi::Object Memory::Init(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "Memory", {
                                                       InstanceMethod("free", &Memory::Free),
                                                       InstanceMethod("copyHostToDevice", &Memory::CopyHostToDevice),
                                                       InstanceMethod("copyDeviceToHost", &Memory::CopyDeviceToHost),
                                                   });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("Memory", func);

  return exports;
}

Napi::Object Memory::New()
{
  return constructor.New({});
}

Memory::Memory(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Memory>(info)
{
}

Napi::Value Memory::Free(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (!validate(cuMemFree(m_ptr), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}

Napi::Value Memory::CopyHostToDevice(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Null();
  }

  if (!info[0].IsArrayBuffer())
  {
    Napi::TypeError::New(env, "Incorrect argument, expected array buffer.").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();

  if (!validate(cuMemcpyHtoD(m_ptr, buf.Data(), buf.ByteLength()), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}

Napi::Value Memory::CopyDeviceToHost(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Null();
  }

  if (!info[0].IsArrayBuffer())
  {
    Napi::TypeError::New(env, "Incorrect argument, expected array buffer.").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();

  if (!validate(cuMemcpyDtoH(buf.Data(), m_ptr, buf.ByteLength()), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}
