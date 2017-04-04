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
struct __ALIGN__(16) HashEntry {
  int3	pos;		   // hash position (lower left corner of SDFBlock))
  int		ptr;	     // pointer into heap to SDFBlock
  uint	offset;		 // offset for collisions

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
};

#endif //VH_CORE_H
