#include "context.hpp"
#include "module.hpp"
#include "device.hpp"
#include "shared.hpp"
#include "memory.hpp"

Napi::FunctionReference Context::constructor;

Napi::Object Context::Init(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "Context", {
                                                        InstanceMethod("moduleLoad", &Context::ModuleLoad),
                                                        InstanceMethod("destroy", &Context::Destroy),
                                                        InstanceMethod("allocMem", &Context::AllocMem),
                                                    });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("Context", func);
  exports.Set("createContext", Napi::Function::New(env, CreateContext));

  return exports;
}

Napi::Value Context::CreateContext(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  Device *device = Napi::ObjectWrap<Device>::Unwrap(info[0].As<Napi::Object>());

  // Create the context
  Napi::Object obj = constructor.New({});
  Context *context = Napi::ObjectWrap<Context>::Unwrap(obj);

  if (!validate(cuCtxCreate(&context->m_context, 0, device->m_device), env))
  {
    return env.Undefined();
  }

  return obj;
}

Context::Context(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Context>(info)
{
}

Napi::Value Context::Destroy(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (!validate(cuCtxDestroy(this->m_context), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}

Napi::Value Context::ModuleLoad(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();
  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::String arg0 = info[0].As<Napi::String>();

  Napi::Object obj = Module::New();
  Module *module = Napi::ObjectWrap<Module>::Unwrap(obj);

  const std::string filename = arg0.Utf8Value();

  if (!validate(cuModuleLoad(&module->m_module, filename.data()), env))
  {
    return env.Undefined();
  }

  return obj;
}

Napi::Value Context::AllocMem(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Null();
  }

  int32_t size = info[0].As<Napi::Number>().Int32Value();

  Napi::Object obj = Memory::New();
  Memory *memory = Napi::ObjectWrap<Memory>::Unwrap(obj);

  if (!validate(cuCtxSetCurrent(m_context), env))
  {
    return env.Undefined();
  }

  if (!validate(cuMemAlloc(&memory->m_ptr, size), env))
  {
    return env.Undefined();
  }

  return obj;
}
