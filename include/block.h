//
// Created by wei on 17-5-21.
//

#ifndef VH_BLOCK_H
#define VH_BLOCK_H

#include "common.h"
#include <helper_math.h>

struct __ALIGN__(4) Voxel {
  float   sdf;		// signed distance function
  uchar3	color;	// color
  uchar	  weight;	// accumulated sdf weight

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
};

struct __ALIGN__(4) MeshCube {
  static const int kTrianglePerCube = 5;

  // TODO(wei): Possible memory optimizations:
  // 1. vertex_ptr point to int3 on shared memory
  // 2. triangle_ptr point to linked list on shared memory
  /// Point to 3 valid vertex indices
  int  vertex_ptrs[3];
  int  triangle_ptr[kTrianglePerCube];
  int  cube_index;

  __device__
  void Clear() {
    vertex_ptrs[0] = -1;
    vertex_ptrs[1] = -1;
    vertex_ptrs[2] = -1;
    for (int i = 0; i < kTrianglePerCube; ++i)
      triangle_ptr[i] = -1;
    cube_index = 0;
  }
};

/// Block
/// Typically Block is a 8x8x8 voxel cluster
struct __ALIGN__(4) VoxelBlock {
  Voxel    voxels[BLOCK_SIZE];
  MeshCube cubes [BLOCK_SIZE];

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

  __device__
  void Update(int i, const Voxel& update) {

    Voxel& in = voxels[i];
    float3 c_in     = make_float3(in.color.x, in.color.y, in.color.z);
    float3 c_update = make_float3(update.color.x, update.color.y, update.color.z);
    float3 c_out = 0.5f * c_in + 0.5f * c_update;

    uchar3 color = make_uchar3(c_out.x + 0.5f, c_out.y + 0.5f, c_out.z + 0.5f);

    in.color.x = color.x, in.color.y = color.y, in.color.z = color.z;
    in.sdf = (in.sdf * (float)in.weight + update.sdf * (float)update.weight)
             / ((float)in.weight + (float)update.weight);
    in.weight = min(255, (uint)in.weight + (uint)update.weight);
  }
};

typedef VoxelBlock* VoxelBlocksGPU;

class VoxelBlocks {
private:
  VoxelBlocksGPU gpu_data_;
  uint block_count_;

  void Alloc(uint block_count);
  void Free();

public:
  VoxelBlocks();
  ~VoxelBlocks();

  void Reset();
  void Resize(uint block_count);

  VoxelBlocksGPU& gpu_data() {
    return gpu_data_;
  }
};

#endif //VOXEL_HASHING_BLOCK_H
