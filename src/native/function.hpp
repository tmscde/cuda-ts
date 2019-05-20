#ifndef FUNCTION_H
#define FUNCTION_H

#include <napi.h>
#include <cuda.h>

class Function : public Napi::ObjectWrap<Function> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New();

  Function(const Napi::CallbackInfo& info);

  CUfunction m_function;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value LaunchKernel(const Napi::CallbackInfo& info);
};

#endif
