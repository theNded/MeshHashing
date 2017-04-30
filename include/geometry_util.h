//
// Created by wei on 17-4-2.
//

#ifndef VOXEL_HASHING_POSITION_CONVERTER_H
#define VOXEL_HASHING_POSITION_CONVERTER_H

#include "common.h"
#include "params.h"

extern __constant__ SDFParams kSDFParams;

///////////////////////////////////////////////////
/// Affected scale: Map (HashTable)
/// Transforms between world, voxel, and block coordinate systems
/// Semantic: A pos To B pos; A, B in {world, voxel, block}
/// float is only used to do interpolation
__device__
inline float3 WorldToVoxelf(const float3& world_pos) {
  return world_pos / kSDFParams.voxel_size;
}
__device__
inline int3 WorldToVoxeli(const float3& world_pos) {
  const float3 p = world_pos / kSDFParams.voxel_size;
  return make_int3(p + make_float3(sign(p)) * 0.5f);
}

__device__
inline int3 VoxelToBlock(int3 voxel_pos) {
  if (voxel_pos.x < 0) voxel_pos.x -= SDF_BLOCK_SIZE-1;
  if (voxel_pos.y < 0) voxel_pos.y -= SDF_BLOCK_SIZE-1;
  if (voxel_pos.z < 0) voxel_pos.z -= SDF_BLOCK_SIZE-1;

  return make_int3(
          voxel_pos.x / SDF_BLOCK_SIZE,
          voxel_pos.y / SDF_BLOCK_SIZE,
          voxel_pos.z / SDF_BLOCK_SIZE);
}

/// Corner voxel with smallest xyz
__device__
inline int3 BlockToVoxel(const int3& block_pos) {
  return block_pos * SDF_BLOCK_SIZE;
}

__device__
inline float3 VoxelToWorld(const int3& voxel_pos) {
  return make_float3(voxel_pos) * kSDFParams.voxel_size;
}

__device__
inline float3 BlockToWorld(const int3& block_pos) {
  return VoxelToWorld(BlockToVoxel(block_pos));
}

__device__
inline int3 WorldToBlock(const float3& world_pos) {
  return VoxelToBlock(WorldToVoxeli(world_pos));
}

/////////////////////////////////////////////
/// Transforms between coordinates and indices
/// Idx means local idx inside a block \in [0, 511]
__device__
inline uint3 IdxToVoxelLocalPos(uint idx) {
  uint x = idx % SDF_BLOCK_SIZE;
  uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
  uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
  return make_uint3(x, y, z);
}

/// Computes the linearized index of a local virtual voxel pos; pos \in [0;7]^3
__device__
inline uint VoxelLocalPosToIdx(const int3& voxel_local_pos) {
  return voxel_local_pos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE +
         voxel_local_pos.y * SDF_BLOCK_SIZE +
         voxel_local_pos.x;
}

__device__
inline int VoxelPosToIdx(const int3& voxel_pos) {
  int3 voxel_local_pos = make_int3(
          voxel_pos.x % SDF_BLOCK_SIZE,
          voxel_pos.y % SDF_BLOCK_SIZE,
          voxel_pos.z % SDF_BLOCK_SIZE);

  if (voxel_local_pos.x < 0) voxel_local_pos.x += SDF_BLOCK_SIZE;
  if (voxel_local_pos.y < 0) voxel_local_pos.y += SDF_BLOCK_SIZE;
  if (voxel_local_pos.z < 0) voxel_local_pos.z += SDF_BLOCK_SIZE;

  return VoxelLocalPosToIdx(voxel_local_pos);
}

__device__
inline int WorldPosToIdx(const float3& world_pos) {
  int3 voxel_pos = WorldToVoxeli(world_pos);
  return VoxelPosToIdx(voxel_pos);
}

///////////////////////////////////////////////////////////////
/// Affected scale: Map (HashTable) and Sensor
/// Projections and reprojections
/// Between the Camera coordinate system and the image plane
/// Projection
__device__
static inline float2 CameraProjectToImagef(const float3& camera_pos,
                                           float fx, float fy,
                                           float cx, float cy)	{
  return make_float2(camera_pos.x * fx / camera_pos.z + cx,
                     camera_pos.y * fy / camera_pos.z + cy);
}

__device__
static inline int2 CameraProjectToImagei(const float3& camera_pos,
                                         float fx, float fy,
                                         float cx, float cy)	{
  float2 uv = CameraProjectToImagef(camera_pos, fx, fy, cx, cy);
  return make_int2(uv + make_float2(0.5f, 0.5f));
}

/// R^3 -> [0, 1]^3
/// maybe used for rendering
__device__
static inline float NormalizeDepth(float z, float min_depth, float max_depth)	{
  return (z - min_depth) / (max_depth - min_depth);
}

///////////////////////////////////////////////////////////////
// Screen to Camera (depth in meters)
///////////////////////////////////////////////////////////////
/// R^2 -> R^3
__device__
static inline float3 ImageReprojectToCamera(uint ux, uint uy, float depth,
                                            float fx, float fy,
                                            float cx, float cy)	{
  const float x = ((float)ux - cx) / fx;
  const float y = ((float)uy - cy) / fy;
  return make_float3(depth * x, depth * y, depth);
}

///////////////////////////////////////////////////////////////
// RenderScreen to Camera -- ATTENTION ASSUMES [1,0]-Z range!!!!
///////////////////////////////////////////////////////////////
/// [0, 1]^3 -> R^3
__device__ /// Normalize
static inline float DenormalizeDepth(float z, float min_depth, float max_depth) {
  return z * (max_depth - min_depth) + min_depth;
}

__device__
static inline bool IsInCameraFrustumApprox(const float4x4& c_T_w,
                                           const float3& world_pos,
                                           const SensorParams& sensor_params) {
  float3 camera_pos = c_T_w * world_pos;
  float2 uv = CameraProjectToImagef(camera_pos,
                                    sensor_params.fx, sensor_params.fy,
                                    sensor_params.cx, sensor_params.cy);
  float3 normalized_p = make_float3(
          (2.0f*uv.x - (sensor_params.width- 1.0f))/(sensor_params.width- 1.0f),
          ((sensor_params.height-1.0f) - 2.0f*uv.y)/(sensor_params.height-1.0f),
          NormalizeDepth(camera_pos.z,
                         sensor_params.min_depth_range,
                         sensor_params.max_depth_range));

  normalized_p *= 0.95;
  return !(normalized_p.x < -1.0f || normalized_p.x > 1.0f
           || normalized_p.y < -1.0f || normalized_p.y > 1.0f
           || normalized_p.z < 0.0f || normalized_p.z > 1.0f);
}

//! returns the truncation of the SDF for a given distance value
__device__
static inline float truncate_distance(float z) {
  return kSDFParams.truncation_distance
         + kSDFParams.truncation_distance_scale * z;
}

// TODO(wei): a better implementation?
__device__
static inline bool IsBlockInCameraFrustum(float4x4 c_T_w, const int3& block_pos,
                                          const SensorParams &sensor_params) {
  float3 world_pos = VoxelToWorld(BlockToVoxel(block_pos))
                     + kSDFParams.voxel_size * 0.5f * (SDF_BLOCK_SIZE - 1.0f);
  return IsInCameraFrustumApprox(c_T_w, world_pos, sensor_params);
}

#endif //VOXEL_HASHING_POSITION_CONVERTER_H
