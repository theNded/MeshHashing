//
// Created by wei on 17-10-22.
//

#include "mapping/allocate.h"

__global__
void AllocBlockArrayKernel(HashTable   hash_table,
                           SensorData  sensor_data,
                           SensorParams sensor_params,
                           float4x4     w_T_c,
                           GeometryHelper geometry_helper) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sensor_params.width || y >= sensor_params.height)
    return;

  /// TODO(wei): change it here
  /// 1. Get observed data
  float depth = tex2D<float>(sensor_data.depth_texture, x, y);
  if (depth == MINF || depth == 0.0f
      || depth >= geometry_helper.sdf_upper_bound)
    return;

  float truncation = geometry_helper.truncate_distance(depth);
  float near_depth = fminf(geometry_helper.sdf_upper_bound, depth - truncation);
  float far_depth = fminf(geometry_helper.sdf_upper_bound, depth + truncation);
  if (near_depth >= far_depth) return;

  float3 camera_pos_near = geometry_helper.ImageReprojectToCamera(x, y, near_depth,
                                                            sensor_params.fx, sensor_params.fy,
                                                            sensor_params.cx, sensor_params.cy);
  float3 camera_pos_far  = geometry_helper.ImageReprojectToCamera(x, y, far_depth,
                                                            sensor_params.fx, sensor_params.fy,
                                                            sensor_params.cx, sensor_params.cy);

  /// 2. Set range where blocks are allocated
  float3 world_pos_near  = w_T_c * camera_pos_near;
  float3 world_pos_far   = w_T_c * camera_pos_far;
  float3 world_ray_dir = normalize(world_pos_far - world_pos_near);

  int3 block_pos_near = geometry_helper.WorldToBlock(world_pos_near);
  int3 block_pos_far  = geometry_helper.WorldToBlock(world_pos_far);
  float3 block_step = make_float3(sign(world_ray_dir));

  /// 3. Init zig-zag steps
  float3 world_pos_nearest_voxel_center
      = geometry_helper.BlockToWorld(block_pos_near + make_int3(clamp(block_step, 0.0, 1.0f)))
        - 0.5f * geometry_helper.voxel_size;
  float3 t = (world_pos_nearest_voxel_center - world_pos_near) / world_ray_dir;
  float3 dt = (block_step * BLOCK_SIDE_LENGTH * geometry_helper.voxel_size) / world_ray_dir;
  int3 block_pos_bound = make_int3(make_float3(block_pos_far) + block_step);

  if (world_ray_dir.x == 0.0f) {
    t.x = PINF;
    dt.x = PINF;
  }
  if (world_ray_dir.y == 0.0f) {
    t.y = PINF;
    dt.y = PINF;
  }
  if (world_ray_dir.z == 0.0f) {
    t.z = PINF;
    dt.z = PINF;
  }

  int3 block_pos_curr = block_pos_near;
  /// 4. Go a zig-zag path to ensure all voxels are visited
  const uint kMaxIterTime = 1024;
#pragma unroll 1
  for (uint iter = 0; iter < kMaxIterTime; ++iter) {
    if (geometry_helper.IsBlockInCameraFrustum(
        w_T_c.getInverse(),
        block_pos_curr,
        sensor_params)) {
      /// Disable streaming at current
      // && !isSDFBlockStreamedOut(idCurrentVoxel, hash_table, is_streamed_mask)) {
      hash_table.AllocEntry(block_pos_curr);
    }

    // Traverse voxel grid
    if (t.x < t.y && t.x < t.z) {
      block_pos_curr.x += block_step.x;
      if (block_pos_curr.x == block_pos_bound.x) return;
      t.x += dt.x;
    } else if (t.y < t.z) {
      block_pos_curr.y += block_step.y;
      if (block_pos_curr.y == block_pos_bound.y) return;
      t.y += dt.y;
    } else {
      block_pos_curr.z += block_step.z;
      if (block_pos_curr.z == block_pos_bound.z) return;
      t.z += dt.z;
    }
  }
}

void AllocBlockArray(
    HashTable& hash_table,
    Sensor& sensor,
    GeometryHelper& geometry_helper
) {
  hash_table.ResetMutexes();

  const uint threads_per_block = 8;
  const dim3 grid_size((sensor.sensor_params().width + threads_per_block - 1)
                       /threads_per_block,
                       (sensor.sensor_params().height + threads_per_block - 1)
                       /threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  AllocBlockArrayKernel<<<grid_size, block_size>>>(
      hash_table,
      sensor.data(),
      sensor.sensor_params(), sensor.wTc(),
      geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}
