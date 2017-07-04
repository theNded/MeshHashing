//
// Created by wei on 17-5-21.
//

#ifndef VH_BLOCK_H
#define VH_BLOCK_H

#include "common.h"
#include <iostream>
#include <helper_math.h>


/// c------  c: center of a cube, it is placed at the corner for easier
/// |     |     vertex value accessing
/// |  v  |  v: center of a voxel, it is naturally at the center of a grid
/// |     |
/// -------
/// Statistics typically reserved for Voxels
struct __ALIGN__(4) Stat {
  float laplacian;
  float entropy;
  float duration;

  __device__
  void Clear() {
    laplacian = 0;
    entropy = 0;
    duration = 0;
  }
};

#define STATS

struct __ALIGN__(8) Voxel {
  float2 ssdf;    // signed distance function
  uchar2 sweight;  // accumulated sdf weight
  uchar3 color;  // color

#ifdef STATS
  Stat   stats;
#endif

  // TODO(wei): Notice, when the size exceeds long long, change this
//  __device__
//  void operator=(const struct Voxel& v) {
//    ((long long*)this)[0] = ((const long long*)&v)[0];
//  }

  __device__
  void Clear() {
    ssdf = make_float2(0.0f, 0.0f);
    sweight = make_uchar2(0, 0);
    color = make_uchar3(0, 0, 0);
  }

  __device__
  float sdf() {
    if (sweight.x + sweight.y == 0) return 0;
    return (ssdf.x * sweight.x + ssdf.y * sweight.y) / (sweight.x + sweight.y);
  }

  __device__
  uchar weight() {
    return (sweight.x + sweight.y);
  }

  __device__
  float entropy() {
    float wp = sweight.x;// * exp(- ssdf.x);
    float wn = sweight.y;// * exp(- ssdf.y);
    float r = wp / (wp + wn);
    if (sweight.x == 0 || sweight.y == 0) return 0;
    return -(r * log(r) + (1 - r) * log(1 - r));
  }

  __device__
  void Update(const Voxel &delta) {
    float3 c_prev = make_float3(color.x, color.y, color.z);
    float3 c_delta = make_float3(delta.color.x, delta.color.y, delta.color.z);
    float3 c_curr = 0.5f * c_prev + 0.5f * c_delta;
    color = make_uchar3(c_curr.x + 0.5f, c_curr.y + 0.5f, c_curr.z + 0.5f);

    ssdf = (ssdf * make_float2(sweight)
            + delta.ssdf * make_float2(delta.sweight))
           / (make_float2(sweight) + make_float2(delta.sweight));
    float2 sweightf = make_float2(sweight) + make_float2(delta.sweight);
    float factor = 255.0f / (sweightf.x + sweightf.y);
    factor = fminf(factor, 1.0);
    sweight = make_uchar2((uchar) (factor * sweightf.x),
                          (uchar) (factor * sweightf.y));

    if (sweight.x == 0) ssdf.x = 0;
    if (sweight.y == 0) ssdf.y = 0;
  }
};

struct __ALIGN__(4) Cube {
  static const int kVerticesPerCube = 3;
  static const int kMaxTrianglesPerCube = 5;

  // TODO(wei): Possible memory optimizations:
  // 1. vertex_ptr:    a @int point to @int3 on shared memory
  // 2. triangle_ptrs: a @int point to linked list on shared memory
  /// Point to 3 valid vertex indices
  int vertex_ptrs[kVerticesPerCube];
  int vertex_mutexes[kVerticesPerCube];
  int triangle_ptrs[kMaxTrianglesPerCube];
  short curr_index, prev_index;

  __device__
  void ResetMutexes() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < kVerticesPerCube; ++i) {
      vertex_mutexes[i] = FREE_PTR;
    }
  }

  __device__
  void Clear() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < kVerticesPerCube; ++i) {
      vertex_ptrs[i] = FREE_PTR;
      vertex_mutexes[i] = FREE_PTR;
    }

#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < kMaxTrianglesPerCube; ++i) {
      triangle_ptrs[i] = FREE_PTR;
    }

    curr_index = 0;
    prev_index = 0;
  }
};

/// Typically Block is a 8x8x8 voxel cluster
struct __ALIGN__(8) Block {
  Voxel voxels[BLOCK_SIZE];
  Cube  cubes[BLOCK_SIZE];

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

typedef Block *BlocksGPU;

class Blocks {
private:
  BlocksGPU gpu_data_;
  uint block_count_;

  void Alloc(uint block_count);

  void Free();

public:
  Blocks();

  ~Blocks();

  void Reset();

  void Resize(uint block_count);

  BlocksGPU &gpu_data() {
    return gpu_data_;
  }
};

#endif //VOXEL_HASHING_BLOCK_H
