#include "event.hpp"
#include "shared.hpp"
#include "stream.hpp"

Napi::FunctionReference Event::constructor;

Napi::Object Event::Init(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "Event", {
                                                      InstanceMethod("destroy", &Event::Destroy),
                                                      InstanceMethod("record", &Event::Record),
                                                      InstanceMethod("synchronize", &Event::Synchronize),
                                                  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("Event", func);
  exports.Set("createEvent", Napi::Function::New(env, &Event::CreateEvent));

  return exports;
}

Napi::Value Event::CreateEvent(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Object obj = constructor.New({});
  Event *event = Napi::ObjectWrap<Event>::Unwrap(obj);

  Napi::Number arg0 = info[0].As<Napi::Number>();
  unsigned int flags = arg0.Int32Value();

  if (!validate(cuEventCreate(&event->m_event, flags), env))
  {
    return env.Undefined();
  }

  return obj;
}

Event::Event(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Event>(info)
{
}

Napi::Value Event::Destroy(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (!validate(cuEventDestroy(m_event), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}

Napi::Value Event::Record(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  CUstream stream = Stream::GetStream(env, info[0]);

  if (!validate(cuEventRecord(m_event, stream), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}

Napi::Value Event::Synchronize(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (!validate(cuEventSynchronize(m_event), env))
  {
    return env.Undefined();
  }

  return env.Undefined();
}
