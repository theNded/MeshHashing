/// Input depth image as texture
/// Easier interpolation

#include "sensor.h"
#include "mapper.h"
#include "hash_table_gpu.h"

#define PINF  __int_as_float(0x7f800000)

/// Refer to sensor.cu
extern texture<float, cudaTextureType2D, cudaReadModeElementType> depth_texture;
extern texture<float4, cudaTextureType2D, cudaReadModeElementType> color_texture;

/// Kernel functions
__global__
void IntegrateCudaKernel(HashTableGPU<Block> hash_table, SensorData sensor_data, float4x4 c_T_w) {
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
  float depth = tex2D(depth_texture, image_pos.x, image_pos.y);
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
    float4 color = tex2D(color_texture, image_pos.x, image_pos.y);
    delta.color = make_uchar3(255 * color.x, 255 * color.y, 255 * color.z);
  } else {
    delta.color = make_uchar3(0, 255, 0);
  }

  hash_table.values[entry.ptr].Update(local_idx, delta);
}

/// Member functions
Mapper::Mapper() {}
Mapper::~Mapper(){}

void Mapper::Integrate(Map *map, Sensor* sensor, uint *is_streamed_mask) {

  //make the rigid transform available on the GPU
  //map->hash_table().updateParams(map->hash_params());
  /// seems OK

  //allocate all hash blocks which are corresponding to depth map entries
  AllocBlocks(map, sensor);
  /// DIFFERENT: is_streamed_mask now empty
  /// seems OK now, supported by MATLAB scatter3

  //generate a linear hash array with only occupied entries
  map->CompactHashEntries(sensor->c_T_w());
  /// seems OK, supported by MATLAB scatter3

  //volumetrically integrate the depth data into the depth SDFBlocks
  IntegrateDepthMap(map, sensor);
  /// cuda kernel launching ok
  /// seems ok according to CUDA output

  map->Recycle();

  map->frame_count() ++;
}

__host__
void Mapper::IntegrateDepthMap(Map* map, Sensor *sensor) {
  const uint threads_per_block = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;

  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count, map->hash_table().compacted_hash_entry_counter,
                  sizeof(uint), cudaMemcpyDeviceToHost));
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  if (occupied_block_count <= 0) return;

  IntegrateCudaKernel << <grid_size, block_size >> >(map->hash_table(), sensor->sensor_data(), sensor->c_T_w());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}


__global__
void AllocBlocksKernel(HashTableGPU<Block> hash_table, SensorData sensor_data,
                       float4x4 w_T_c, const uint* is_streamed_mask) {
  const HashParams &hash_params = kHashParams;
  const SensorParams &sensor_params = kSensorParams;

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sensor_params.width || y >= sensor_params.height)
    return;

  /// TODO(wei): change it here
  /// 1. Get observed data
  float depth = tex2D(depth_texture, x, y);
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

__host__
void Mapper::AllocBlocks(Map* map, Sensor* sensor) {
  const uint threads_per_block = 8;
  const dim3 grid_size((sensor->sensor_params().width + threads_per_block - 1)/threads_per_block,
                       (sensor->sensor_params().height + threads_per_block - 1)/threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  AllocBlocksKernel<<<grid_size, block_size>>>(map->hash_table(), sensor->sensor_data(), sensor->w_T_c(), NULL);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}
