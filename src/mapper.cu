/// Input depth image as texture
/// Easier interpolation

#include "hash_table.h"
#include "sensor_data.h"

#define PINF  __int_as_float(0x7f800000)
#define T_PER_BLOCK 8

//////////
/// Bind texture
texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef;
texture<float4, cudaTextureType2D, cudaReadModeElementType> colorTextureRef;
__host__
void BindSensorDataToTextureCudaHost(const SensorData& sensor_data) {
  checkCudaErrors(cudaBindTextureToArray(depthTextureRef, sensor_data.depth_array_, sensor_data.h_depthChannelDesc));
  checkCudaErrors(cudaBindTextureToArray(colorTextureRef, sensor_data.color_array_, sensor_data.h_colorChannelDesc));
  depthTextureRef.filterMode = cudaFilterModePoint;
  colorTextureRef.filterMode = cudaFilterModePoint;
}

//////////
/// Integrate depth map
/// Private compcated_hash_entries, blocks
__global__
void IntegrateCudaKernel(HashTable hash_table, SensorData sensor_data, float4x4 c_T_w) {
  const HashParams &hash_params = kHashParams;
  const SensorParams &sensor_params = kSensorParams;

  //TODO check if we should load this in shared memory

  /// 1. Select voxel
  const HashEntry &entry = hash_table.compacted_hash_entries[blockIdx.x];
  int3 voxel_base_pos = BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(IdxToVoxelLocalPos(local_idx));

  /// 2. Project to camera
  float3 world_pos = VoxelToWorld(voxel_pos);
  float3 camera_pos = c_T_w * world_pos;
  uint2 image_pos = make_uint2(CameraProjectToImagei(camera_pos));
  if (image_pos.x >= sensor_params.width || image_pos.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float depth = tex2D(depthTextureRef, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= hash_params.sdf_upper_bound)
    return;

  /// 4. Truncate
  float sdf = depth - camera_pos.z;
  float truncation = truncate_distance(depth);
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
  delta.weight = max(hash_params.weight_sample * 1.5f * (1.0f - NormalizeDepth(depth)), 1.0f);
  if (sensor_data.color_image_) {
    float4 color = tex2D(colorTextureRef, image_pos.x, image_pos.y);
    delta.color = make_uchar3(255 * color.x, 255 * color.y, 255 * color.z);
  } else {
    delta.color = make_uchar3(0, 255, 0);
  }

  uint idx = entry.ptr + local_idx;
  hash_table.UpdateVoxel(hash_table.blocks[idx], delta);
}


__host__
void IntegrateCudaHost(HashTable& hash_table, const HashParams& hash_params,
                       const SensorData& sensor_data, const SensorParams& sensor_params,
                       float4x4 c_T_w) {
  const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;

  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count, hash_table.compacted_hash_entry_counter,
                  sizeof(uint), cudaMemcpyDeviceToHost));
  const dim3 gridSize(occupied_block_count, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  if (occupied_block_count > 0) {	//this guard is important if there is no depth in the current frame (i.e., no blocks were allocated)
    IntegrateCudaKernel << <gridSize, blockSize >> >(hash_table, sensor_data, c_T_w);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}


//////////
/// Alloc blocks in the frustum around observed 3D points
/// Public AllocBlock
__global__
void AllocBlocksKernel(HashTable hash_table, SensorData sensor_data,
                       float4x4 w_T_c, const unsigned int* is_streamed_mask) {
  const HashParams &hash_params = kHashParams;
  const SensorParams &sensor_params = kSensorParams;

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sensor_params.width || y >= sensor_params.height)
    return;

  /// TODO(wei): change it here
  /// 1. Get observed data
  float depth = tex2D(depthTextureRef, x, y);
  if (depth == MINF || depth == 0.0f
      || depth >= hash_params.sdf_upper_bound)
    return;

  float truncation = truncate_distance(depth);
  float near_depth = min(hash_params.sdf_upper_bound, depth - truncation);
  float far_depth = min(hash_params.sdf_upper_bound, depth + truncation);
  if (near_depth >= far_depth) return;

  float3 camera_pos_near = ImageReprojectToCamera(x, y, near_depth);
  float3 camera_pos_far  = ImageReprojectToCamera(x, y, far_depth);

  /// 2. Set range where blocks are allocated
  float3 world_pos_near  = w_T_c * camera_pos_near;
  float3 world_pos_far   = w_T_c * camera_pos_far;
  float3 world_ray_dir = normalize(world_pos_far - world_pos_near);

  int3 block_pos_near = WorldToBlock(world_pos_near);
  int3 block_pos_far  = WorldToBlock(world_pos_far);
  float3 block_step = make_float3(sign(world_ray_dir));

  /// 3. Init zig-zag steps
  float3 world_pos_nearest_voxel_center
          = BlockToWorld(block_pos_near + make_int3(clamp(block_step, 0.0, 1.0f)))
            - 0.5f * hash_params.voxel_size;
  float3 t = (world_pos_nearest_voxel_center - world_pos_near) / world_ray_dir;
  float3 dt = (block_step * SDF_BLOCK_SIZE * hash_params.voxel_size) / world_ray_dir;
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
    if (IsBlockInCameraFrustum(w_T_c.getInverse(), block_pos_curr)) {
      /// Disable streaming at current
      // && !isSDFBlockStreamedOut(idCurrentVoxel, hash_table, is_streamed_mask)) {
      hash_table.AllocBlock(block_pos_curr);
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

__host__
void AllocBlocksCudaHost(HashTable& hash_table, const HashParams& hash_params,
                         const SensorData& sensor_data, const SensorParams& sensor_params,
                         const float4x4& w_T_c, const unsigned int* is_streamed_mask) {

  const dim3 gridSize((sensor_params.width + T_PER_BLOCK - 1)/T_PER_BLOCK, (sensor_params.height + T_PER_BLOCK - 1)/T_PER_BLOCK);
  const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

  AllocBlocksKernel<<<gridSize, blockSize>>>(hash_table, sensor_data, w_T_c, is_streamed_mask);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}
