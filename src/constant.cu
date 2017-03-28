#include "hash_param.h"
#include "ray_caster_param.h"
#include "sensor_param.h"
#include <helper_cuda.h>

__constant__ HashParams c_hashParams;
void updateConstantHashParams(const HashParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, c_hashParams));
  checkCudaErrors(cudaMemcpyToSymbol(c_hashParams, &params, size, 0, cudaMemcpyHostToDevice));
}

__constant__ DepthCameraParams c_depthCameraParams;
void updateConstantDepthCameraParams(const DepthCameraParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, c_depthCameraParams));
  checkCudaErrors(cudaMemcpyToSymbol(c_depthCameraParams, &params, size, 0, cudaMemcpyHostToDevice));
}

__constant__ RayCastParams c_rayCastParams;
void updateConstantRayCastParams(const RayCastParams& params) {
  size_t size;
  checkCudaErrors(cudaGetSymbolSize(&size, c_rayCastParams));
  checkCudaErrors(cudaMemcpyToSymbol(c_rayCastParams, &params, size, 0, cudaMemcpyHostToDevice));
}