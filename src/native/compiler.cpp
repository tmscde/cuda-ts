#include <nvrtc.h>
#include "compiler.hpp"
#include "shared.hpp"

#define validateNvrtc(result, env) __validateNvrtcResult(result, __FILE__, __LINE__, env)

extern inline bool __validateNvrtcResult(nvrtcResult result, const char *file, const int line, Napi::Env env)
{
  if (NVRTC_SUCCESS != result)
  {
    char message[2048];
    // const char **errorMessage = NULL;
    // cuGetErrorName(err, errorMessage);
    sprintf(message,
            "NVRTC API error = %04d from file <%s>, line %i.\n",
            result /*, *errorMessage*/, file, line);
    Napi::TypeError::New(env, message).ThrowAsJavaScriptException();
    return false;
  }
  return true;
}

Napi::Value CompileCuToPtx(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  // Get the cu-code
  Napi::String arg0 = info[0].As<Napi::String>();

  // Null terminated cu-code string
  const char *cuCode = arg0.Utf8Value().c_str();

  nvrtcProgram prog;

  if (!validateNvrtc(nvrtcCreateProgram(&prog, cuCode, "program.cu", 0, NULL, NULL), env))
  {
    return env.Undefined();
  }

  int numCompileOptions = 0;
  char *compileParams[1];

  nvrtcResult compilerRes = nvrtcCompileProgram(prog, numCompileOptions, compileParams);

  // dump log
  size_t logSize;
  if (!validateNvrtc(nvrtcGetProgramLogSize(prog, &logSize), env))
  {
    return env.Undefined();
  }

  char *log = reinterpret_cast<char *>(malloc(sizeof(char) * logSize + 1));
  if (!validateNvrtc(nvrtcGetProgramLog(prog, log), env))
  {
    return env.Undefined();
  }

  log[logSize] = '\x0';

  if (strlen(log) >= 2)
  {
    fprintf(stdout, "\n compilation log ---\n%s\n end log ---\n", log);
  }

  if (!validateNvrtc(compilerRes, env))
  {
    return env.Undefined();
  }

  size_t ptxSize;
  if (!validateNvrtc(nvrtcGetPTXSize(prog, &ptxSize), env))
  {
    return env.Undefined();
  }

  char *ptx = reinterpret_cast<char *>(malloc(sizeof(char) * ptxSize));
  if (!validateNvrtc(nvrtcGetPTX(prog, ptx), env))
  {
    return env.Undefined();
  }

  if (!validateNvrtc(nvrtcDestroyProgram(&prog), env))
  {
    return env.Undefined();
  }

  return Napi::ArrayBuffer::New(env, ptx, ptxSize);
}

Napi::Object InitCompiler(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  exports.Set("compileCuToPtx", Napi::Function::New(env, CompileCuToPtx));
  return exports;
}
