//
// Created by wei on 17-3-12.
//

/// Core data structures for VoxelHashing
/// Lowest level structs
/// Header both used for .cu and .cc

#ifndef VH_CORE_H
#define VH_CORE_H

#include "common.h"

/// HashEntry
struct __ALIGN__(8) HashEntry {
  int3	pos;		   // hash position (lower left corner of SDFBlock))
  int		ptr;	     // pointer into heap to SDFBlock
  uint	offset;		 // offset for collisions

  // uint padding

  __device__ void operator=(const struct HashEntry& e) {
    ((long long*)this)[0] = ((const long long*)&e)[0];
    ((long long*)this)[1] = ((const long long*)&e)[1];
    ((int*)this)[4] = ((const int*)&e)[4];
  }
};

/// Block: Cluster of 256 Voxels

/// Voxel
struct __ALIGN__(8) Voxel {
  float   sdf;		// signed distance function
  uchar3	color;	// color
  uchar	  weight;	// accumulated sdf weight

  __device__ void operator=(const struct Voxel& v) {
    ((long long*)this)[0] = ((const long long*)&v)[0];
  }
  __device__ void Clear() {
    sdf = 0.0;
    color = make_uchar3(0, 0, 0);
    weight = 0;
  }
};

/// Block
/// Typically Block is an array of Voxels
/// TODO: a more reasonable wrapper for accessing
struct __ALIGN__(8) Block {
  Voxel voxels[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];

  __device__ Voxel& operator() (int i) {
    return voxels[i];
  }

  __device__ void Clear() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < 512; ++i) {
      voxels[i].Clear();
    }
  }

  __device__ void Update(int i, const Voxel& update) {
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
#endif //VH_CORE_H
