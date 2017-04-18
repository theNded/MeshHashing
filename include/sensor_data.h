//
// Created by wei on 17-3-16.
//

#ifndef MRF_VH_DEPTH_CAMERA_H
#define MRF_VH_DEPTH_CAMERA_H

#include "common.h"
#include "sensor_param.h"

#include <helper_cuda.h>
#include <matrix.h>

/// constant.cu
extern __constant__ SensorParams kSensorParams;
extern void SetConstantSensorParams(const SensorParams& params);

struct SensorData {
  ///////////////
  // Host part //
  ///////////////
  __device__ __host__
  SensorData() {
    depth_image_ = NULL;
    color_image_ = NULL;
    depth_array_ = NULL;
    color_array_ = NULL;
  }

  __host__
  void Alloc(const SensorParams& params) {
    checkCudaErrors(cudaMalloc(&depth_image_, sizeof(float) * params.width * params.height));
    checkCudaErrors(cudaMalloc(&color_image_, sizeof(float4) * params.width * params.height));

    h_depthChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&depth_array_, &h_depthChannelDesc, params.width, params.height));
    h_colorChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&color_array_, &h_colorChannelDesc, params.width, params.height));
  }

  __host__
  void Free() {
    if (depth_image_) checkCudaErrors(cudaFree(depth_image_));
    if (color_image_) checkCudaErrors(cudaFree(color_image_));
    if (depth_array_) checkCudaErrors(cudaFreeArray(depth_array_));
    if (color_array_) checkCudaErrors(cudaFreeArray(color_array_));

    depth_image_ = NULL;
    color_image_ = NULL;
    depth_array_ = NULL;
    color_array_ = NULL;
  }

  /// Raw data
  float*		depth_image_;
  float4*		color_image_;

  /// Texture-binded data
  cudaArray*	depth_array_;
  cudaArray*	color_array_;
  cudaChannelFormatDesc h_depthChannelDesc;
  cudaChannelFormatDesc h_colorChannelDesc;
};

#endif //MRF_VH_DEPTH_CAMERA_H
