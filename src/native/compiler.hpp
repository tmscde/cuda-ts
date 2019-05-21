#ifndef COMPILER_H
#define COMPILER_H

#include <napi.h>
#include <cuda.h>

Napi::Object InitCompiler(Napi::Env env, Napi::Object exports);

#endif
