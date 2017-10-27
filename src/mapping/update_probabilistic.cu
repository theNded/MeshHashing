//
// Created by wei on 17-10-25.
//

#include <device_launch_parameters.h>
#include "update_probabilistic.h"

#include "core/block_array.h"
#include "mapping/update_simple.h"
#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/spatial_query.h"

////////////////////
/// Device code
////////////////////
__global__
void BuildSensorDataEquationKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    SensorData sensor_data,
    SensorParams params,
    float4x4 cTw,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    SensorLinearEquations linear_equations
) {

  //TODO check if we should load this in shared memory (entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.IdxToVoxelLocalPos(local_idx));

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];
  /// 2. Project to camera
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);
  float3 camera_pos = cTw * world_pos;
  uint2 image_pos = make_uint2(
      geometry_helper.CameraProjectToImagei(camera_pos,
                                            params.fx, params.fy,
                                            params.cx, params.cy));
  if (image_pos.x >= params.width || image_pos.y >= params.height)
    return;

  /// 3. Find correspondent depth observation
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= geometry_helper.sdf_upper_bound)
    return;
  float sdf = depth - camera_pos.z;
  float truncation = geometry_helper.truncate_distance(depth);
  if (fabs(sdf) > truncation) // outlier
    return;

  /// 4. Build linear system
  float3 sample_c = geometry_helper.ImageReprojectToCamera(image_pos.x, image_pos.y,
                                                           depth,
                                                           params.fx, params.fy,
                                                           params.cx, params.cy);
  float3 sample_w = cTw.getInverse() * sample_c;
  int pixel_idx = image_pos.y * params.width + image_pos.x;

  /// Solve (I + \sum \lambda nn^T + ... )x = (dp + \sum \lambda nn^Tv)
  float normalized_depth = geometry_helper.NormalizeDepth(depth,
                                                          params.min_depth_range,
                                                          params.max_depth_range);

  float3x3 A = float3x3::getZeroMatrix();
  float3 b = make_float3(0);

  for (int i = 0; i < N_VERTEX; ++i) {
    if (this_voxel.vertex_ptrs[i] > 0) {
      Vertex vtx = mesh.vertex(this_voxel.vertex_ptrs[i]);
      float3 v = vtx.pos;
      float3 n = vtx.normal;
      float3x3 nnT = float3x3(n.x * n.x, n.x * n.y, n.x * n.z,
                              n.y * n.x, n.y * n.y, n.y * n.z,
                              n.z * n.x, n.z * n.y, n.z * n.z);
      // TODO: use Nguyen's model
      float lambda = 1;
      A = A + nnT * lambda;
      b = b + nnT * v * lambda;
    }
  }
  linear_equations.atomicAddfloat3x3(pixel_idx, A);
  linear_equations.atomicAddfloat3(pixel_idx, b);
}

__global__
void UpdateBlocksBayesianKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    SensorData sensor_data,
    SensorLinearEquations linear_equations,
    SensorParams sensor_params,
    float4x4 cTw,
    HashTable hash_table,
    GeometryHelper geometry_helper) {

  //TODO check if we should load this in shared memory (entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.IdxToVoxelLocalPos(local_idx));

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];
  /// 2. Project to camera
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);
  float3 camera_pos = cTw * world_pos;
  uint2 image_pos = make_uint2(
      geometry_helper.CameraProjectToImagei(camera_pos,
                                            sensor_params.fx, sensor_params.fy,
                                            sensor_params.cx, sensor_params.cy));
  if (image_pos.x >= sensor_params.width
      || image_pos.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= geometry_helper.sdf_upper_bound)
    return;

  int pixel_idx = image_pos.y * sensor_params.width + image_pos.x;
  float3 sample_pos_cam = linear_equations.b[pixel_idx];

  float sdf = sample_pos_cam.z - camera_pos.z;
  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float weight = (1.0f - normalized_depth);
  float truncation = geometry_helper.truncate_distance(depth);
  if (sdf <= -truncation)
    return;
  if (sdf >= 0.0f) {
    sdf = fminf(truncation, sdf);
  } else {
    sdf = fmaxf(-truncation, sdf);
  }

  /// 5. Update
  Voxel delta;
  delta.sdf = sdf;
  delta.weight = weight;

  if (sensor_data.color_data) {
    float4 color = tex2D<float4>(sensor_data.color_texture, image_pos.x, image_pos.y);
    delta.color = make_uchar3(255 * color.x, 255 * color.y, 255 * color.z);
  } else {
    delta.color = make_uchar3(0, 255, 0);
  }
  this_voxel.Update(delta);
}

void BuildSensorDataEquation(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh,
    Sensor &sensor,
    HashTable &hash_table,
    GeometryHelper &geometry_helper,
    SensorLinearEquations &linear_equations
) {
  const uint threads_per_block = BLOCK_SIZE;

  uint compacted_entry_count = candidate_entries.count();
  if (compacted_entry_count <= 0)
    return;

  const dim3 grid_size(compacted_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);
  BuildSensorDataEquationKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          mesh,
          sensor.data(),
          sensor.sensor_params(),
          sensor.cTw(),
          hash_table,
          geometry_helper,
          linear_equations);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void UpdateBlocksBayesian(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Sensor &sensor,
    SensorLinearEquations &linear_equations,
    HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  const uint threads_per_block = BLOCK_SIZE;

  uint compacted_entry_count = candidate_entries.count();
  if (compacted_entry_count <= 0)
    return;

  const dim3 grid_size(compacted_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);
  UpdateBlocksBayesianKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          sensor.data(),
          linear_equations,
          sensor.sensor_params(),
          sensor.cTw(),
          hash_table,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}