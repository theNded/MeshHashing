#include "params.h"

#include <helper_cuda.h>

__constant__ SDFParams kSDFParams;
void SetConstantSDFParams(const SDFParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, kSDFParams));
  checkCudaErrors(cudaMemcpyToSymbol(kSDFParams, &params,
                                     size, 0,
                                     cudaMemcpyHostToDevice));
}