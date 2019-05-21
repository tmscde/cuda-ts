#include "function.hpp"
#include "memory.hpp"
#include "shared.hpp"

Napi::FunctionReference Function::constructor;

Napi::Object Function::Init(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "Function", {
                                                         InstanceMethod("launchKernel", &Function::LaunchKernel),
                                                     });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("Function", func);

  return exports;
}

Napi::Object Function::New()
{
  return constructor.New({});
}

Function::Function(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Function>(info)
{
}

Napi::Value Function::LaunchKernel(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Array array = info[0].As<Napi::Array>();

  std::vector<CUdeviceptr *> input(array.Length());
  for (int i = 0; i < (int)array.Length(); i++)
  {
    input[i] = &Napi::ObjectWrap<Memory>::Unwrap(array.Get(i).As<Napi::Object>())->m_ptr;
  }

  // Synchronization is not needed as we're using stream 0
  if (!validate(cuLaunchKernel(m_function,
                               2, 1, 1, // Grid dimensions (block count)
                               1, 1, 1, // Block dimensions (thread count)
                               0,       // Shared mem bytes
                               0,       // Stream
                               (void **)input.data(),
                               NULL // Extra
                               ),
                env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}
