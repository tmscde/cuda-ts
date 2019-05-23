#ifndef CONTEXT_H
#define CONTEXT_H

#include <napi.h>
#include <cuda.h>

class Context : public Napi::ObjectWrap<Context>
{
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Value CreateContext(const Napi::CallbackInfo &info);

  Context(const Napi::CallbackInfo &info);

private:
  static Napi::FunctionReference constructor;

  Napi::Value Destroy(const Napi::CallbackInfo &info);
  Napi::Value ModuleLoad(const Napi::CallbackInfo &info);
  Napi::Value ModuleLoadData(const Napi::CallbackInfo &info);
  Napi::Value AllocMem(const Napi::CallbackInfo &info);

  CUcontext m_context;
};

#endif
