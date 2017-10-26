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
      float lambda = this_voxel.weight;
      A = A + nnT * lambda;
      b = b + nnT * v * lambda;
    }
  }
  linear_equations.atomicAddfloat3x3(pixel_idx, A);
  linear_equations.atomicAddfloat3(pixel_idx, b);
}

__global__
void SolveSensorDataEquationKernel(
    SensorLinearEquations linear_equations,
    const SensorData sensor_data,
    const SensorParams params,
    float4x4 wTc,
    GeometryHelper geometry_helper
) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int idx = y * params.width + x;
  float depth = tex2D<float>(sensor_data.depth_texture, x, y);
  if (depth == MINF || depth == 0.0f || depth > geometry_helper.sdf_upper_bound)
    return;

  float3 input_pos = geometry_helper.ImageReprojectToCamera(x, y, depth,
                                                            params.fx, params.fy,
                                                            params.cx, params.cy);
  float3 predict_pos = (float3x3::getIdentity() + linear_equations.A[idx]).getInverse()
                       * (input_pos + linear_equations.b[idx]);
  // low support -> outlier
  printf("(%f %f %f) -> (%f %f %f)\n",
         input_pos.x, input_pos.y, input_pos.z,
         predict_pos.x, predict_pos.y, predict_pos.z);
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

void SolveSensorDataEquation(
    SensorLinearEquations& linear_equations,
    Sensor& sensor,
    GeometryHelper& geometry_helper
) {
  const SensorParams& params = sensor.sensor_params();
  const SensorData& data = sensor.data();
  uint width = params.width;
  uint height = params.height;

  const int threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1)/threads_per_block,
                       (height + threads_per_block - 1)/threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  SolveSensorDataEquationKernel <<<grid_size, block_size>>> (
      linear_equations,
      data,
      params,
      sensor.wTc(),
      geometry_helper);
}