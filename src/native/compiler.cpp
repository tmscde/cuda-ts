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

void freePtx(Napi::Env env, void *ptr)
{
  free(ptr);
}

Napi::Value CompileCuToPtx(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();

  if (info.Length() != 1)
  {
    Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string cuCode = info[0].As<Napi::String>().Utf8Value();
  nvrtcProgram prog;

  if (!validateNvrtc(nvrtcCreateProgram(&prog, cuCode.c_str(), "program.cu", 0, NULL, NULL), env))
  {
    return env.Undefined();
  }

  int numCompileOptions = 0;
  char *compileParams[1] = {NULL};

  nvrtcResult compilerRes = nvrtcCompileProgram(prog, numCompileOptions, compileParams);
  if (compilerRes != NVRTC_SUCCESS)
  {
    size_t logSize;
    if (nvrtcGetProgramLogSize(prog, &logSize) == NVRTC_SUCCESS)
    {
      char *log = new char[logSize];
      if (nvrtcGetProgramLog(prog, log) == NVRTC_SUCCESS)
      {
        fprintf(stdout, "\n compilation log ---\n%s\n end log ---\n", log);
      }
      delete[] log;
    }
    nvrtcDestroyProgram(&prog);
    validateNvrtc(compilerRes, env);
    return env.Undefined();
  }

  size_t ptxSize;
  if (!validateNvrtc(nvrtcGetPTXSize(prog, &ptxSize), env))
  {
    nvrtcDestroyProgram(&prog);
    return env.Undefined();
  }

  char *ptx = reinterpret_cast<char *>(malloc(sizeof(char) * ptxSize));
  if (!validateNvrtc(nvrtcGetPTX(prog, ptx), env))
  {
    nvrtcDestroyProgram(&prog);
    return env.Undefined();
  }

  if (!validateNvrtc(nvrtcDestroyProgram(&prog), env))
  {
    return env.Undefined();
  }

  Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(env, ptx, ptxSize, &freePtx);

  return buffer;
}

Napi::Object InitCompiler(Napi::Env env, Napi::Object exports)
{
  Napi::HandleScope scope(env);

  exports.Set("compileCuToPtx", Napi::Function::New(env, CompileCuToPtx));
  return exports;
}
