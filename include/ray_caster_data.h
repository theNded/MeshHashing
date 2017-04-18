//
// Created by wei on 17-3-17.
//

#ifndef MRF_VH_RAY_CASTER_DATA_H
#define MRF_VH_RAY_CASTER_DATA_H

#include "common.h"
#include <helper_cuda.h>

#include "hash_table.h"
#include "ray_caster_param.h"

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

/// constant.cu
extern __constant__ RayCasterParams kRayCasterParams;
extern void SetConstantRayCasterParams(const RayCasterParams &params);

struct RayCastSample {
  float sdf;
  float alpha;
  uint weight;
};

struct RayCasterData {
  ///////////////
  // Host part //
  ///////////////

  __device__ __host__
  RayCasterData() {
    depth_image_ = NULL;
    vertex_image_ = NULL;
    normal_image_ = NULL;
    color_image_ = NULL;
  }

  __host__
  void Alloc(const RayCasterParams &params) {
    checkCudaErrors(cudaMalloc(&depth_image_, sizeof(float) * params.m_width * params.m_height));
    checkCudaErrors(cudaMalloc(&vertex_image_, sizeof(float4) * params.m_width * params.m_height));
    checkCudaErrors(cudaMalloc(&normal_image_, sizeof(float4) * params.m_width * params.m_height));
    checkCudaErrors(cudaMalloc(&color_image_, sizeof(float4) * params.m_width * params.m_height));
  }

  __host__
  void Free() {
    checkCudaErrors(cudaFree(depth_image_));
    checkCudaErrors(cudaFree(vertex_image_));
    checkCudaErrors(cudaFree(normal_image_));
    checkCudaErrors(cudaFree(color_image_));
  }

  /// WRITE in ray_caster,
  /// specifically in traverseCoarseGridSimpleSampleAll
  float *depth_image_;
  float4 *vertex_image_;
  float4 *normal_image_;
  float4 *color_image_;
};


#endif //MRF_VH_RAY_CASTER_DATA_H
