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

  uint  block_count;                // 1000000

  int   block_size;                 // 8 (voxels)

  uint  voxel_count;                // block_count * block_size^3
  float	voxel_size;                 // 0.004 (m)

  float	truncation_distance_scale;  // 0.01 (m / m)
  float	truncation_distance;        // 0.02 (m)
  float	sdf_upper_bound;            // 4.0 (m)

  uint  weight_sample;              // 10,  TODO(wei): change it dynamically!
  uint  weight_upper_bound;         // 255
};
#endif //VH_HASH_PARAM_H
