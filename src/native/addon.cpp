#include <napi.h>
#include <curand_kernel.h>
#include "device.hpp"
#include "context.hpp"
#include "module.hpp"
#include "memory.hpp"
#include "function.hpp"
#include "stream.hpp"
#include "event.hpp"
#include "compiler.hpp"

Napi::Value GetSizeofCurandState(const Napi::CallbackInfo &info)
{
  return Napi::Number::New(info.Env(), sizeof(curandState));
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports)
{
  cuInit(0);

  Device::Init(env, exports);
  Context::Init(env, exports);
  Module::Init(env, exports);
  Memory::Init(env, exports);
  Function::Init(env, exports);
  Stream::Init(env, exports);
  Event::Init(env, exports);
  InitCompiler(env, exports);

  Napi::HandleScope scope(env);

  exports.Set("getSizeofCurandState", Napi::Function::New(env, GetSizeofCurandState));

  return exports;
}

NODE_API_MODULE(cuda, InitAll)
