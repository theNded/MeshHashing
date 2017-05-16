/// Input depth image as texture
/// Easier interpolation
#include <device_launch_parameters.h>

#include <helper_cuda.h>
#include <glog/logging.h>
#include "sensor.h"
#include "fuser.h"
#include "hash_table_gpu.h"

#define PINF  __int_as_float(0x7f800000)

/// Refer to sensor.cu
extern texture<float, cudaTextureType2D, cudaReadModeElementType> depth_texture;
extern texture<float4, cudaTextureType2D, cudaReadModeElementType> color_texture;

/// Kernel functions
template <typename T>
__global__
void CollectTargetBlocksKernel(HashTableGPU<T> hash_table,
                              SensorParams sensor_params, // K && min/max depth
                              float4x4 c_T_w) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < *hash_table.entry_count) {
    if (hash_table.hash_entries[idx].ptr != FREE_ENTRY) {
      if (IsBlockInCameraFrustum(c_T_w, hash_table.hash_entries[idx].pos,
                                 sensor_params)) {
        addr_local = atomicAdd(&local_counter, 1);
      }
    }
  }

  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(hash_table.compacted_hash_entry_counter, local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    hash_table.compacted_hash_entries[addr] = hash_table.hash_entries[idx];
  }
}

__global__
void UpdateBlocksKernel(HashTableGPU<VoxelBlock> map_table,
                        SensorData sensor_data,
                        SensorParams sensor_params,
                        float4x4 c_T_w) {

  //TODO check if we should load this in shared memory (compacted_entries)
  /// 1. Select voxel
  const HashEntry &entry = map_table.compacted_hash_entries[blockIdx.x];
  int3 voxel_base_pos = BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(IdxToVoxelLocalPos(local_idx));

  /// 2. Project to camera
  float3 world_pos = VoxelToWorld(voxel_pos);
  float3 camera_pos = c_T_w * world_pos;
  uint2 image_pos = make_uint2(CameraProjectToImagei(camera_pos,
                                                     sensor_params.fx, sensor_params.fy,
                                                     sensor_params.cx, sensor_params.cy));
  if (image_pos.x >= sensor_params.width
      || image_pos.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float depth = tex2D(depth_texture, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= kSDFParams.sdf_upper_bound)
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
  delta.weight = max(kSDFParams.weight_sample * 1.5f *
                             (1.0f - NormalizeDepth(depth,
                                                    sensor_params.min_depth_range,
                                                    sensor_params.max_depth_range)),
                     1.0f);
  if (sensor_data.color_image) {
    float4 color = tex2D(color_texture, image_pos.x, image_pos.y);
    delta.color = make_uchar3(255 * color.x, 255 * color.y, 255 * color.z);
  } else {
    delta.color = make_uchar3(0, 255, 0);
  }

  map_table.values[entry.ptr].Update(local_idx, delta);
}

__global__
void AllocBlocksKernel(HashTableGPU<VoxelBlock> map_table,
                       SensorData sensor_data,
                       SensorParams sensor_params,
                       float4x4 w_T_c, const uint* is_streamed_mask) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sensor_params.width || y >= sensor_params.height)
    return;

  /// TODO(wei): change it here
  /// 1. Get observed data
  float depth = tex2D(depth_texture, x, y);
  if (depth == MINF || depth == 0.0f
      || depth >= kSDFParams.sdf_upper_bound)
    return;

  float truncation = truncate_distance(depth);
  float near_depth = fminf(kSDFParams.sdf_upper_bound, depth - truncation);
  float far_depth = fminf(kSDFParams.sdf_upper_bound, depth + truncation);
  if (near_depth >= far_depth) return;

  float3 camera_pos_near = ImageReprojectToCamera(x, y, near_depth,
                                                  sensor_params.fx, sensor_params.fy,
                                                  sensor_params.cx, sensor_params.cy);
  float3 camera_pos_far  = ImageReprojectToCamera(x, y, far_depth,
                                                  sensor_params.fx, sensor_params.fy,
                                                  sensor_params.cx, sensor_params.cy);

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
            - 0.5f * kSDFParams.voxel_size;
  float3 t = (world_pos_nearest_voxel_center - world_pos_near) / world_ray_dir;
  float3 dt = (block_step * BLOCK_SIDE_LENGTH * kSDFParams.voxel_size) / world_ray_dir;
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
    if (IsBlockInCameraFrustum(w_T_c.getInverse(), block_pos_curr, sensor_params)) {
      /// Disable streaming at current
      // && !isSDFBlockStreamedOut(idCurrentVoxel, hash_table, is_streamed_mask)) {
      map_table.AllocEntry(block_pos_curr);
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

/// Member functions (CPU code)
Fuser::Fuser() {}
Fuser::~Fuser(){}

void Fuser::Integrate(Map *map, Sensor* sensor, uint *is_streamed_mask) {

  //make the rigid transform available on the GPU
  //map->gpu_data().updateParams(map->hash_params());
  /// seems OK

  //allocate all hash blocks which are corresponding to depth map entries
  AllocBlocks(map, sensor);

  /// DIFFERENT: is_streamed_mask now empty
  /// seems OK now, supported by MATLAB scatter3

  //generate a linear hash array with only occupied entries
  CollectTargetBlocks(map, sensor);
  /// seems OK, supported by MATLAB scatter3

  //volumetrically integrate the depth data into the depth SDFBlocks
  UpdateBlocks(map, sensor);
  /// cuda kernel launching ok
  /// seems ok according to CUDA output

  map->Recycle();

  map->frame_count() ++;
}

/// Member functions (CPU calling GPU kernels)
void Fuser::AllocBlocks(Map* map, Sensor* sensor) {
  map->hash_table().ResetMutexes();

  const uint threads_per_block = 8;
  const dim3 grid_size((sensor->sensor_params().width + threads_per_block - 1)
                       /threads_per_block,
                       (sensor->sensor_params().height + threads_per_block - 1)
                       /threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  AllocBlocksKernel<<<grid_size, block_size>>>(map->gpu_data(),
          sensor->sensor_data(), sensor->sensor_params(), sensor->w_T_c(), NULL);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void Fuser::CollectTargetBlocks(Map* map, Sensor *sensor){
  const uint threads_per_block = 256;
  uint res = 0;

  uint entry_count;
  checkCudaErrors(cudaMemcpy(&entry_count, map->gpu_data().entry_count,
                             sizeof(uint), cudaMemcpyDeviceToHost));

  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  checkCudaErrors(cudaMemset(map->gpu_data().compacted_hash_entry_counter,
                             0, sizeof(int)));
  CollectTargetBlocksKernel<<<grid_size, block_size >>>(map->gpu_data(),
          sensor->sensor_params(), sensor->c_T_w());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(&res, map->gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  LOG(INFO) << "Block count in view frustum: " << res;
}

void Fuser::UpdateBlocks(Map* map, Sensor *sensor) {
  const uint threads_per_block = BLOCK_SIZE;

  uint compacted_entry_count = map->hash_table().compacted_entry_count();
  if (compacted_entry_count <= 0)
    return;

  const dim3 grid_size(compacted_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);
  UpdateBlocksKernel <<<grid_size, block_size>>>(map->gpu_data(),
          sensor->sensor_data(), sensor->sensor_params(), sensor->c_T_w());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}