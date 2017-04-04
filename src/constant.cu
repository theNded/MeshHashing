#include "hash_param.h"
#include "ray_caster_param.h"
#include "sensor_param.h"
#include <helper_cuda.h>

__constant__ HashParams kHashParams;
void UpdateConstantHashParams(const HashParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, kHashParams));
  checkCudaErrors(cudaMemcpyToSymbol(kHashParams, &params, size, 0, cudaMemcpyHostToDevice));
}

__constant__ SensorParams kSensorParams;
void UpdateConstantSensorParams(const SensorParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, kSensorParams));
  checkCudaErrors(cudaMemcpyToSymbol(kSensorParams, &params, size, 0, cudaMemcpyHostToDevice));
}

__constant__ RayCastParams kRayCastParams;
void UpdateConstantRayCastParams(const RayCastParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, kRayCastParams));
  checkCudaErrors(cudaMemcpyToSymbol(kRayCastParams, &params, size, 0, cudaMemcpyHostToDevice));
}