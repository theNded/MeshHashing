#include "params.h"

#include <helper_cuda.h>

__constant__ SDFParams kSDFParams;
void SetConstantSDFParams(const SDFParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, kSDFParams));
  checkCudaErrors(cudaMemcpyToSymbol(kSDFParams, &params, size, 0, cudaMemcpyHostToDevice));
}

__constant__ SensorParams kSensorParams;
void SetConstantSensorParams(const SensorParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, kSensorParams));
  checkCudaErrors(cudaMemcpyToSymbol(kSensorParams, &params, size, 0, cudaMemcpyHostToDevice));
}

__constant__ RayCasterParams kRayCasterParams;
void SetConstantRayCasterParams(const RayCasterParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, kRayCasterParams));
  checkCudaErrors(cudaMemcpyToSymbol(kRayCasterParams, &params, size, 0, cudaMemcpyHostToDevice));
}