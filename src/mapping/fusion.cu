#include <device_launch_parameters.h>

#include "core/block_array.h"
#include "mapping/fusion.h"
#include "engine/mapping_engine.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/gradient.h"


////////////////////
/// class MappingEngine - integrate sensor data
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void UpdateBlockArrayKernel(EntryArray candidate_entries,
                        HashTable        hash_table,
                        BlockArray          blocks,
                        Mesh             mesh,
                        SensorDataGPU       sensor_data,
                        SensorParams        sensor_params,
                        float4x4            c_T_w,
                        CoordinateConverter converter) {

  //TODO check if we should load this in shared memory (entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = converter.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(converter.IdxToVoxelLocalPos(local_idx));

  Voxel& this_voxel = blocks[entry.ptr].voxels[local_idx];
  /// 2. Project to camera
  float3 world_pos = converter.VoxelToWorld(voxel_pos);
  float3 camera_pos = c_T_w * world_pos;
  uint2 image_pos = make_uint2(
          converter.CameraProjectToImagei(camera_pos,
                                          sensor_params.fx, sensor_params.fy,
                                          sensor_params.cx, sensor_params.cy));
  if (image_pos.x >= sensor_params.width
      || image_pos.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= converter.sdf_upper_bound)
    return;

  /// 4. SDF computation
  float3 dp = converter.ImageReprojectToCamera(image_pos.x, image_pos.y, depth,
      sensor_params.fx, sensor_params.fy, sensor_params.cx, sensor_params.cy);
  float3 dpw = c_T_w.getInverse() * dp;

  /// Solve (I + \sum \lambda nn^T + ... )x = (dp + \sum \lambda nn^Tv)
  float3x3 A = float3x3::getIdentity();
  float3   b = dpw;
  float wd = (1.0f - converter.NormalizeDepth(depth,
                                   sensor_params.min_depth_range,
                                   sensor_params.max_depth_range));
  float wn = 0.5f;
  bool addition = false;
  for (int i = 0; i < N_VERTEX; ++i) {
    if (this_voxel.vertex_ptrs[i] > 0) {
      addition = true;
      Vertex vtx = mesh.vertex(this_voxel.vertex_ptrs[i]);
      float3 v = vtx.pos;
      float3 n = vtx.normal;
      wn += dot(c_T_w * n, normalize(-dp));
      float3x3 nnT = float3x3(n.x*n.x, n.x*n.y, n.x*n.z,
                              n.y*n.x, n.y*n.y, n.y*n.z,
                              n.z*n.x, n.z*n.y, n.z*n.z);

      float dist = length(dpw - v);
      float wdist = dist / converter.voxel_size;
      float ww = expf(- wdist*wdist);
      A = A + nnT * ww;
      b = b + nnT * v * ww;
    }
  }

  // Best estimation for dp
  if (addition) {
    dpw = A.getInverse() * b;
  }
  dp = c_T_w * dpw;
  //float3 np = normalize(-dp);

  //printf("%f %f %f\n", np.x, np.y, np.z)

  //float sdf = dot(normalize(-dp), camera_pos - dp);
  float sdf = depth - camera_pos.z;
  //uchar weight = (uchar)fmax(1.0f, kSDFParams.weight_sample * wn * wd);

  float weight = (uchar)fmax(converter.weight_sample * 1.5f *
                     (1.0f - converter.NormalizeDepth(depth,
                                            sensor_params.min_depth_range,
                                            sensor_params.max_depth_range)),
                     1.0f);
  float truncation = converter.truncate_distance(depth);
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

  if (sensor_data.color_image) {
    float4 color = tex2D<float4>(sensor_data.color_texture, image_pos.x, image_pos.y);
    delta.color = make_uchar3(255 * color.x, 255 * color.y, 255 * color.z);
  } else {
    delta.color = make_uchar3(0, 255, 0);
  }
  this_voxel.Update(delta);
}

__global__
void AllocBlockArrayKernel(HashTable   hash_table,
                       SensorDataGPU  sensor_data,
                       SensorParams   sensor_params,
                       float4x4       w_T_c,
                       const uint* is_streamed_mask,
                       CoordinateConverter converter) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sensor_params.width || y >= sensor_params.height)
    return;

  /// TODO(wei): change it here
  /// 1. Get observed data
  float depth = tex2D<float>(sensor_data.depth_texture, x, y);
  if (depth == MINF || depth == 0.0f
      || depth >= converter.sdf_upper_bound)
    return;

  float truncation = converter.truncate_distance(depth);
  float near_depth = fminf(converter.sdf_upper_bound, depth - truncation);
  float far_depth = fminf(converter.sdf_upper_bound, depth + truncation);
  if (near_depth >= far_depth) return;

  float3 camera_pos_near = converter.ImageReprojectToCamera(x, y, near_depth,
                                                  sensor_params.fx, sensor_params.fy,
                                                  sensor_params.cx, sensor_params.cy);
  float3 camera_pos_far  = converter.ImageReprojectToCamera(x, y, far_depth,
                                                  sensor_params.fx, sensor_params.fy,
                                                  sensor_params.cx, sensor_params.cy);

  /// 2. Set range where blocks are allocated
  float3 world_pos_near  = w_T_c * camera_pos_near;
  float3 world_pos_far   = w_T_c * camera_pos_far;
  float3 world_ray_dir = normalize(world_pos_far - world_pos_near);

  int3 block_pos_near = converter.WorldToBlock(world_pos_near);
  int3 block_pos_far  = converter.WorldToBlock(world_pos_far);
  float3 block_step = make_float3(sign(world_ray_dir));

  /// 3. Init zig-zag steps
  float3 world_pos_nearest_voxel_center
          = converter.BlockToWorld(block_pos_near + make_int3(clamp(block_step, 0.0, 1.0f)))
            - 0.5f * converter.voxel_size;
  float3 t = (world_pos_nearest_voxel_center - world_pos_near) / world_ray_dir;
  float3 dt = (block_step * BLOCK_SIDE_LENGTH * converter.voxel_size) / world_ray_dir;
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
    if (converter.IsBlockInCameraFrustum(w_T_c.getInverse(), block_pos_curr, sensor_params)) {
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

void AllocBlockArray(HashTable& hash_table, Sensor& sensor, CoordinateConverter& converter) {
  hash_table.ResetMutexes();

  const uint threads_per_block = 8;
  const dim3 grid_size((sensor.sensor_params().width + threads_per_block - 1)
                       /threads_per_block,
                       (sensor.sensor_params().height + threads_per_block - 1)
                       /threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  AllocBlockArrayKernel<<<grid_size, block_size>>>(
          hash_table,
          sensor.gpu_memory(),
          sensor.sensor_params(), sensor.w_T_c(),
          NULL,
              converter);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void UpdateBlockArray(EntryArray& candidate_entries,
                      HashTable&  hash_table,
                      BlockArray& blocks,
                      Mesh& mesh,
                      Sensor &sensor,
                      CoordinateConverter& converter) {
  const uint threads_per_block = BLOCK_SIZE;

  uint compacted_entry_count = candidate_entries.count();
  if (compacted_entry_count <= 0)
    return;

  const dim3 grid_size(compacted_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);
  UpdateBlockArrayKernel <<<grid_size, block_size>>>(
          candidate_entries,
          hash_table,
          blocks,
          mesh,
          sensor.gpu_memory(),
          sensor.sensor_params(), sensor.c_T_w(),
              converter);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}