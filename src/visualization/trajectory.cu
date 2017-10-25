//
// Created by wei on 17-10-24.
//

#include <extern/cuda/helper_cuda.h>
#include <glog/logging.h>
#include "trajectory.h"

Trajectory::Trajectory(uint max_vertex_count) {
  Init(max_vertex_count);
}

// Free()
void Trajectory::Init(uint max_vertex_count) {
  max_vertex_count_ = max_vertex_count;
  checkCudaErrors(cudaMalloc(&vertices_, sizeof(float3) * max_vertex_count));
}

void Trajectory::Free() {
  checkCudaErrors(cudaFree(vertices_));
}

void Trajectory::AddPose(float4x4 wTc) {
  float3 curr_position = make_float3(wTc.m14, wTc.m24, wTc.m34);
  if (vertex_count_ == -1) {
    vertex_count_ = 0;
  } else {
    checkCudaErrors(cudaMemcpy(vertices_ + vertex_count_,
                               &prev_position_,
                               sizeof(float3),
                               cudaMemcpyHostToDevice));
    vertex_count_++;
    checkCudaErrors(cudaMemcpy(vertices_ + vertex_count_,
                               &curr_position,
                               sizeof(float3),
                               cudaMemcpyHostToDevice));
    vertex_count_++;

    frustum_ = MakeFrustum(wTc);
    for (int i = 0; i < frustum_.size(); ++i) {
      checkCudaErrors(cudaMemcpy(vertices_ + vertex_count_ + i,
                                 &frustum_[i],
                                 sizeof(float3),
                                 cudaMemcpyHostToDevice));
    }
  }
  prev_position_ = curr_position;
}

std::vector<float3> Trajectory::MakeFrustum(float4x4 wTc) {
  float3 camera_position = make_float3(wTc.m14, wTc.m24, wTc.m34);

  float length = 0.25;

  float4 v04 = wTc * make_float4(length, length, 2*length, 1);
  float4 v14 = wTc * make_float4(length, -length, 2*length, 1);
  float4 v24 = wTc * make_float4(-length, length, 2*length, 1);
  float4 v34 = wTc * make_float4(-length, -length, 2*length, 1);
  float3 v0 = make_float3(v04.x, v04.y, v04.z);
  float3 v1 = make_float3(v14.x, v14.y, v14.z);
  float3 v2 = make_float3(v24.x, v24.y, v24.z);
  float3 v3 = make_float3(v34.x, v34.y, v34.z);

  std::vector<float3> vertices = {camera_position, v0,
                                  camera_position, v1,
                                  camera_position, v2,
                                  camera_position, v3,
                                  v0, v1,
                                  v1, v3,
                                  v3, v2,
                                  v2, v0};
  return vertices;
}