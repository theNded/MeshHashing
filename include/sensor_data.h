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
extern void UpdateConstantSensorParams(const SensorParams& params);

struct SensorData {
  ///////////////
  // Host part //
  ///////////////
  __device__ __host__
  SensorData() {
    d_depthData = NULL;
    d_colorData = NULL;
    d_depthArray = NULL;
    d_colorArray = NULL;
  }

  __host__
  void alloc(const SensorParams& params) {
    checkCudaErrors(cudaMalloc(&d_depthData, sizeof(float) * params.width * params.height));
    checkCudaErrors(cudaMalloc(&d_colorData, sizeof(float4) * params.width * params.height));

    h_depthChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&d_depthArray, &h_depthChannelDesc, params.width, params.height));
    h_colorChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&d_colorArray, &h_colorChannelDesc, params.width, params.height));

    /// Added here
    updateParams(params);
  }

  __host__
  void updateParams(const SensorParams& params) {
    UpdateConstantSensorParams(params);
  }

  __host__
  void Free() {
    if (d_depthData) checkCudaErrors(cudaFree(d_depthData));
    if (d_colorData) checkCudaErrors(cudaFree(d_colorData));
    if (d_depthArray) checkCudaErrors(cudaFreeArray(d_depthArray));
    if (d_colorArray) checkCudaErrors(cudaFreeArray(d_colorArray));

    d_depthData = NULL;
    d_colorData = NULL;
    d_depthArray = NULL;
    d_colorArray = NULL;
  }


  /////////////////
  // Device part //
  /////////////////
  static inline const SensorParams& params() {
    return kSensorParams;
  }



  /// Raw data
  float*		d_depthData;
  float4*		d_colorData;

  /// Texture-binded data
  cudaArray*	d_depthArray;
  cudaArray*	d_colorArray;
  cudaChannelFormatDesc h_depthChannelDesc;
  cudaChannelFormatDesc h_colorChannelDesc;
};

#endif //MRF_VH_DEPTH_CAMERA_H
