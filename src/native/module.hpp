#ifndef MODULE_H
#define MODULE_H

#include <napi.h>
#include <cuda.h>

class Module : public Napi::ObjectWrap<Module> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New();
  Module(const Napi::CallbackInfo& info);

  Napi::Value Unload(const Napi::CallbackInfo& info);

  CUmodule m_module;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value GetFunction(const Napi::CallbackInfo& info);

};

#endif
