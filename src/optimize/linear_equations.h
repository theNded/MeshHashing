//
// Created by wei on 17-10-25.
//

#ifndef MESH_HASHING_LINEAR_LEAST_SQUARES_H
#define MESH_HASHING_LINEAR_LEAST_SQUARES_H

#include "core/common.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"
#include "helper_math.h"
#include "matrix.h"

class SensorLinearEquations {
public:
  // Ax = b for each pixel
  float3x3* A;
  float3*   b;

  void Alloc(int width, int height);
  void Free();
  void Reset();

#ifdef __CUDACC__
  __device__
  void atomicAddfloat3x3(int idx, float3x3& dA) {
#pragma unroll 1
    for (int i = 0; i < 9; ++i) {
      atomicAdd(&(A[idx].entries[i]), dA.entries[i]);
    }
  }
  __device__
  void atomicAddfloat3(int idx, float3& db) {
    atomicAdd(&b[idx].x, db.x);
    atomicAdd(&b[idx].y, db.y);
    atomicAdd(&b[idx].z, db.z);
  }
#endif

private:
  bool is_allocated_on_gpu_ = false;
  int  width_;
  int  height_;
};

void SolveSensorDataEquation(
    SensorLinearEquations& linear_equations,
    Sensor& sensor,
    GeometryHelper& geometry_helper
);

#endif //MESH_HASHING_LINEAR_LEAST_SQUARES_H
