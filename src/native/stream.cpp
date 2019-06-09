#include "stream.hpp"
#include "shared.hpp"
#include <string>

Napi::FunctionReference Stream::constructor;

Napi::Object Stream::Init(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "Stream", {
                                                       InstanceMethod("destroy", &Stream::Destroy),
                                                   });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("Stream", func);
  exports.Set("createStream", Napi::Function::New(env, CreateStream));

  return exports;
}

Napi::Value Stream::CreateStream(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Number arg0 = info[0].As<Napi::Number>();

  Napi::Object obj = constructor.New({});
  Stream *stream = Napi::ObjectWrap<Stream>::Unwrap(obj);
  int flags = arg0.Int32Value();

  if (!validate(cuStreamCreate(&stream->m_stream, flags), env))
  {
    return env.Undefined();
  }

  return obj;
}

Stream::Stream(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Stream>(info)
{
}

Napi::Value Stream::Destroy(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (!validate(cuStreamDestroy(m_stream), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}

CUstream Stream::GetStream(Napi::Env env, Napi::Value value)
{
  CUstream stream = 0;
  if (value.IsNumber())
  {
    if (value.As<Napi::Number>().Int32Value() != 0)
    {
      Napi::TypeError::New(env, "Invalid stream").ThrowAsJavaScriptException();
    }
  }
  else
  {
    stream = (Napi::ObjectWrap<Stream>::Unwrap(value.As<Napi::Object>()))->m_stream;
  }
  return stream;
}