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


// TODO(wei): Notice, when the size exceeds long long, change this
//  __device__
//  void operator=(const struct Voxel& v) {
//    ((long long*)this)[0] = ((const long long*)&v)[0];
//  }

struct __ALIGN__(8) Voxel {
  float  sdf;    // signed distance function
  float  weight;
  uchar3 color;  // color

  int vertex_ptrs[N_VERTEX];
  int vertex_mutexes[N_VERTEX];
  int triangle_ptrs[N_TRIANGLE];

#ifdef STATS
  Stat   stats;
#endif
  short curr_index, prev_index;

  __device__
  void ResetMutexes() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < N_VERTEX; ++i) {
      vertex_mutexes[i] = FREE_PTR;
    }
  }

  __device__
  void ClearSDF() {
    sdf = 0.0f;
    weight = 0;
    color = make_uchar3(0, 0, 0);
  }

  __device__
  void ClearMesh() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < N_VERTEX; ++i) {
      vertex_ptrs[i] = FREE_PTR;
      vertex_mutexes[i] = FREE_PTR;
    }

#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < N_TRIANGLE; ++i) {
      triangle_ptrs[i] = FREE_PTR;
    }

    curr_index = 0;
    prev_index = 0;
  }

  __device__
  void Update(const Voxel &delta) {
    float3 c_prev  = make_float3(color.x, color.y, color.z);
    float3 c_delta = make_float3(delta.color.x, delta.color.y, delta.color.z);
    float3 c_curr  = 0.5f * c_prev + 0.5f * c_delta;
    color = make_uchar3(c_curr.x + 0.5f, c_curr.y + 0.5f, c_curr.z + 0.5f);

    sdf = (sdf * (float)weight + delta.sdf * (float)delta.weight)
          / ((float)weight + (float)delta.weight);
    weight = weight + delta.weight;
  }
};

/// Typically Block is a 8x8x8 voxel cluster
struct __ALIGN__(8) Block {
  Voxel voxels[BLOCK_SIZE];

  __device__
  void Clear() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      voxels[i].ClearSDF();
      voxels[i].ClearMesh();
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
