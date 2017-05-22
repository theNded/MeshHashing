#include "params.h"

#include <helper_cuda.h>

__device__ __constant__ SDFParams kSDFParams;
void SetConstantSDFParams(const SDFParams& params) {
  checkCudaErrors(cudaMemcpyToSymbol(kSDFParams, &params,
                                     sizeof(SDFParams), 0,
                                     cudaMemcpyHostToDevice));
}