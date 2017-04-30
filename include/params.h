//
// Created by wei on 17-3-12.
//

/// Parameters for HashTable
/// Shared in __constant__ form
/// Update it ONLY in hash_table
// TODO(wei): put rigid transform elsewhere

#ifndef VH_HASH_PARAM_H
#define VH_HASH_PARAM_H

#include "common.h"
#include <matrix.h>

struct __ALIGN__(16) HashParams {
  //////////////////////////////////////////////////
  /// Currently used parameters
  uint  bucket_count;               // 500000
  uint  bucket_size;                // 10 (entries)
  uint  entry_count;                // bucket_count * bucket_size
  uint  linked_list_size;           // 7

  uint  value_capacity;                // 1000000

  int   block_size;                 // 8 (voxels)

  uint  voxel_count;                // block_count * block_size^3
  float	voxel_size;                 // 0.004 (m)

  float	truncation_distance_scale;  // 0.01 (m / m)
  float	truncation_distance;        // 0.02 (m)
  float	sdf_upper_bound;            // 4.0 (m)

  uint  weight_sample;              // 10,  TODO(wei): change it dynamically!
  uint  weight_upper_bound;         // 255

  uint3 dummy;
};

struct __ALIGN__(16) SDFParams {
  float	voxel_size;                 // 0.004 (m)

  float	truncation_distance_scale;  // 0.01 (m / m)
  float	truncation_distance;        // 0.02 (m)
  float	sdf_upper_bound;            // 4.0 (m)

  uint  weight_sample;              // 10,  TODO(wei): change it dynamically!
  uint  weight_upper_bound;         // 255

  uint2 padding;
};

struct __ALIGN__(16) RayCasterParams {
  float4x4 intrinsics;               /// Intrinsic matrix
  float4x4 intrinsics_inverse;

  uint width;                /// 640
  uint height;               /// 480

  float min_raycast_depth;
  float max_raycast_depth;
  float raycast_step;                /// 0.8f * SDF_Truncation

  float sample_sdf_threshold;        /// 50.5f * s_rayIncrement
  float sdf_threshold;               /// 50.0f * s_rayIncrement
  bool  enable_gradients;

  uchar3 dummy0;
};

/// We may generate a virtual camera for LiDAR
/// where many points are null
struct __ALIGN__(16) SensorParams {
  float fx;              /// Set manually
  float fy;
  float cx;
  float cy;

  uint width;            /// 640
  uint height;           /// 480

  float min_depth_range; /// 0.5f
  float max_depth_range; /// 5.0f, might need modify for LiDAR
};

#endif //VH_HASH_PARAM_H
