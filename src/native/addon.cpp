#include <napi.h>
#include "device.hpp"
#include "context.hpp"
#include "module.hpp"
#include "memory.hpp"
#include "function.hpp"

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  cuInit(0);

  Device::Init(env, exports);
  Context::Init(env, exports);
  Module::Init(env, exports);
  Memory::Init(env, exports);
  Function::Init(env, exports);

  return exports;
}

NODE_API_MODULE(cuda, InitAll)
