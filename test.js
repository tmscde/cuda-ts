"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const cuda = require("./src/cuda");
cuda.compileCuToPtx(`extern "C" __global__ void add(float* input1, float* input2, float* output) {
    output[blockIdx.x] = input1[blockIdx.x] + input2[blockIdx.x];
  }
`);
//# sourceMappingURL=test.js.map