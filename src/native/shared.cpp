#include "shared.hpp"

extern inline bool __validateCudaResult(CUresult err, const char *file, const int line, Napi::Env env)
{
  if (CUDA_SUCCESS != err)
  {
    char message[2048];
    // const char **errorMessage = NULL;
    // cuGetErrorName(err, errorMessage);
    sprintf(message,
            "CUDA Driver API error = %04d from file <%s>, line %i.\n",
            err /*, *errorMessage*/, file, line);
    Napi::TypeError::New(env, message).ThrowAsJavaScriptException();
    return false;
  }
  return true;
}
