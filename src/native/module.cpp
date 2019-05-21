#include "module.hpp"
#include "shared.hpp"
#include "function.hpp"
#include <string>

Napi::FunctionReference Module::constructor;

Napi::Object Module::Init(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "Module", {
                                                       InstanceMethod("getFunction", &Module::GetFunction),
                                                       InstanceMethod("unload", &Module::Unload),
                                                   });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("Module", func);
  return exports;
}

Napi::Object Module::New()
{
  return constructor.New({});
}

Module::Module(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Module>(info)
{
}

Napi::Value Module::GetFunction(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::String arg0 = info[0].As<Napi::String>();

  Napi::Object obj = Function::New();
  Function *function = Napi::ObjectWrap<Function>::Unwrap(obj);

  const std::string funcName = arg0.Utf8Value();

  if (!validate(cuModuleGetFunction(&function->m_function, m_module, funcName.data()), env))
  {
    return env.Undefined();
  }

  return obj;
}

Napi::Value Module::Unload(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (!validate(cuModuleUnload(m_module), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}
