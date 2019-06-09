#ifndef EVENT_H
#define EVENT_H

#include <napi.h>
#include <cuda.h>

class Event : public Napi::ObjectWrap<Event>
{
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Value CreateEvent(const Napi::CallbackInfo &info);

  Event(const Napi::CallbackInfo &info);

  CUevent m_event;

private:
  static Napi::FunctionReference constructor;

  Napi::Value Record(const Napi::CallbackInfo &info);
  Napi::Value Synchronize(const Napi::CallbackInfo &info);
  // Napi::Value Query(const Napi::CallbackInfo &info);
  // Napi::Value ElapsedTime(const Napi::CallbackInfo &info);
  Napi::Value Destroy(const Napi::CallbackInfo &info);
};

#endif
