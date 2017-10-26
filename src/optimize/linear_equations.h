//
// Created by wei on 17-10-25.
//

#ifndef MESH_HASHING_LINEAR_LEAST_SQUARES_H
#define MESH_HASHING_LINEAR_LEAST_SQUARES_H

#include "core/common.h"
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
      atomicAdd(&(A[i].entries[idx]), dA.entries[i]);
    }
  }
  __device__
  void atomicAddfloat3(int i, float3& db) {
    atomicAdd(&b[i].x, db.x);
    atomicAdd(&b[i].y, db.y);
    atomicAdd(&b[i].z, db.z);
  }
#endif

private:
  bool is_allocated_on_gpu_ = false;
  int  width_;
  int  height_;
};


#endif //MESH_HASHING_LINEAR_LEAST_SQUARES_H
