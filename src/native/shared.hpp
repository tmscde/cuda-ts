#ifndef SHARED_H
#define SHARED_H

#include <stdio.h>
#include <cuda.h>
#include <napi.h>

#define validate(err, env)  __validateCudaResult (err, __FILE__, __LINE__, env)

extern inline bool __validateCudaResult(CUresult err, const char *file, const int line, Napi::Env env);

#endif
