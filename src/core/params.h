//
// Created by wei on 17-3-12.
//

/// Parameters for HashTable
/// Shared in __constant__ form
/// Update it ONLY in hash_table
// TODO(wei): put rigid transform elsewhere

#ifndef VH_PARAMS_H
#define VH_PARAMS_H

#include "core/common.h"
#include <matrix.h>

struct HashParams {
  uint  bucket_count;               // 500000
  uint  bucket_size;                // 10 (entries)

  uint  entry_count;                // bucket_count * bucket_size
  uint  linked_list_size;           // 7

  uint  value_capacity;             // 1000000
};

struct MeshParams {
  uint max_vertex_count;
  uint max_triangle_count;
};

struct __ALIGN__(16) VolumeParams {
  float	voxel_size;                 // 0.004 (m)

  float	truncation_distance_scale;  // 0.01 (m / m)
  float	truncation_distance;        // 0.02 (m)
  float	sdf_upper_bound;            // 4.0 (m)

  uint  weight_sample;              // 10,  TODO(wei): change it dynamically!
  uint  weight_upper_bound;         // 255

  uint2 padding;
};

struct RayCasterParams {
  float  fx;
  float  fy;
  float  cx;
  float  cy;

  uint   width;                /// 640
  uint   height;               /// 480

  float  min_raycast_depth;
  float  max_raycast_depth;
  float  raycast_step;         /// 0.8f * SDF_Truncation

  float  sample_sdf_threshold; /// 50.5f * s_rayIncrement
  float  sdf_threshold;        /// 50.0f * s_rayIncrement
  bool   enable_gradients;
};

/// We may generate a virtual camera for LiDAR
/// where many points are null
struct SensorParams {
  float fx;              /// Set manually
  float fy;
  float cx;
  float cy;

  uint width;            /// 640
  uint height;           /// 480

  float min_depth_range; /// 0.5f
  float max_depth_range; /// 5.0f, might need modify for LiDAR
  float range_factor;    /// 1/5000 for TUM and ICL, 1/1000 for SUN3D
};

/// Just a supersede of argv editing...
struct RuntimeParams {
  int  dataset_type;

  bool free_walk;
  bool ploygon_mode;
  bool mesh_range;
  bool fine_gradient;
  bool render_type;

  bool bounding_box;
  bool ray_casting;

  bool record_video;
  bool save_mesh;

  std::string filename_prefix;
  std::string time_profile;
  std::string memo_profile;

  int run_frames;
};

#endif //VH_PARAMS_H
