#include "hash_param.h"
#include "ray_caster_param.h"
#include "sensor_param.h"
#include <helper_cuda.h>

__constant__ HashParams kHashParams;
void SetConstantHashParams(const HashParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, kHashParams));
  checkCudaErrors(cudaMemcpyToSymbol(kHashParams, &params, size, 0, cudaMemcpyHostToDevice));
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