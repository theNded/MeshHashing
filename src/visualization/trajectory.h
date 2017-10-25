//
// Created by wei on 17-10-24.
//

#ifndef MESH_HASHING_TRAJECTORY_H
#define MESH_HASHING_TRAJECTORY_H

#include <vector>
#include "helper_math.h"
#include "matrix.h"

class Trajectory {
public:
  Trajectory() = default;
  explicit Trajectory(uint max_vertex_count);
  void Init(uint max_vertex_count);
  void Free();

  void AddPose(float4x4 wTc);
  std::vector<float3> MakeFrustum(float4x4 wTc);

  uint vertex_count() {
    return vertex_count_ + frustum_.size();
  }
  float3* vertices() {
    return vertices_;
  }

private:
  uint max_vertex_count_;
  int vertex_count_ = -1;
  float3 prev_position_;
  float3* vertices_;
  std::vector<float3> frustum_;
};


#endif //MESH_HASHING_TRAJECTORY_H
