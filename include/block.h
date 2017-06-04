//
// Created by wei on 17-5-21.
//

#ifndef VH_BLOCK_H
#define VH_BLOCK_H

#include "common.h"
#include <helper_math.h>

/// c------  c: center of a cube, it is placed at the corner for easier
/// |     |     vertex value accessing
/// |  v  |  v: center of a voxel, it is naturally at the center of a grid
/// |     |
/// -------
struct __ALIGN__(4) Voxel {
  float   sdf;		// signed distance function
  uchar3	color;	// color
  uchar	  weight;	// accumulated sdf weight

  // TODO(wei): Notice, when the size exceeds long long, change this
  __device__
  void operator=(const struct Voxel& v) {
    ((long long*)this)[0] = ((const long long*)&v)[0];
  }

  __device__
  void Clear() {
    sdf    = 0.0;
    color  = make_uchar3(0, 0, 0);
    weight = 0;
  }

  __device__
  void Update(const Voxel& delta) {
    float3 c_prev  = make_float3(color.x, color.y, color.z);
    float3 c_delta = make_float3(delta.color.x, delta.color.y, delta.color.z);
    float3 c_curr  = 0.5f * c_prev + 0.5f * c_delta;
    color = make_uchar3(c_curr.x + 0.5f, c_curr.y + 0.5f, c_curr.z + 0.5f);

    sdf = (sdf * (float)weight + delta.sdf * (float)delta.weight)
          / ((float)weight + (float)delta.weight);
    weight = min(255, (uint)weight + (uint)delta.weight);
  }
};

struct __ALIGN__(4) Cube {
  static const int kVerticesPerCube     = 3;
  static const int kMaxTrianglesPerCube = 5;

  // TODO(wei): Possible memory optimizations:
  // 1. vertex_ptr:    a @int point to @int3 on shared memory
  // 2. triangle_ptrs: a @int point to linked list on shared memory
  /// Point to 3 valid vertex indices
  int  vertex_ptrs  [kVerticesPerCube];
  int  triangle_ptrs[kMaxTrianglesPerCube];
  int  cube_index;

  __device__
  void Clear() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < kVerticesPerCube; ++i) {
      vertex_ptrs[i] = -1;
    }

#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < kMaxTrianglesPerCube; ++i) {
      triangle_ptrs[i] = -1;
    }

    cube_index = 0;
  }
};

/// Typically Block is a 8x8x8 voxel cluster
struct __ALIGN__(4) Block {
  Voxel voxels[BLOCK_SIZE];
  Cube  cubes [BLOCK_SIZE];

  __device__
  void Clear() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      voxels[i].Clear();
      cubes[i].Clear();
    }
  }
};

typedef Block* BlocksGPU;

class Blocks {
private:
  BlocksGPU gpu_data_;
  uint      block_count_;

  void Alloc(uint block_count);
  void Free();

public:
  Blocks();
  ~Blocks();

  void Reset();
  void Resize(uint block_count);

  BlocksGPU& gpu_data() {
    return gpu_data_;
  }
};

#endif //VOXEL_HASHING_BLOCK_H
