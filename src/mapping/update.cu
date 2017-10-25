#include <device_launch_parameters.h>

#include "core/block_array.h"
#include "mapping/update.h"
#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/gradient.h"

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
                        GeometryHelper geoemtry_helper) {

  //TODO check if we should load this in shared memory (entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geoemtry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geoemtry_helper.IdxToVoxelLocalPos(local_idx));

  Voxel& this_voxel = blocks[entry.ptr].voxels[local_idx];
  /// 2. Project to camera
  float3 world_pos = geoemtry_helper.VoxelToWorld(voxel_pos);
  float3 camera_pos = c_T_w * world_pos;
  uint2 image_pos = make_uint2(
          geoemtry_helper.CameraProjectToImagei(camera_pos,
                                          sensor_params.fx, sensor_params.fy,
                                          sensor_params.cx, sensor_params.cy));
  if (image_pos.x >= sensor_params.width
      || image_pos.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= geoemtry_helper.sdf_upper_bound)
    return;

  /// 4. SDF computation
  float3 dp = geoemtry_helper.ImageReprojectToCamera(image_pos.x, image_pos.y, depth,
      sensor_params.fx, sensor_params.fy, sensor_params.cx, sensor_params.cy);
  float3 dpw = c_T_w.getInverse() * dp;

  /// Solve (I + \sum \lambda nn^T + ... )x = (dp + \sum \lambda nn^Tv)
  float3x3 A = float3x3::getIdentity();
  float3   b = dpw;
  float wd = (1.0f - geoemtry_helper.NormalizeDepth(depth,
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
      float wdist = dist / geoemtry_helper.voxel_size;
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
  //uchar weight = (uchar)fmax(1.0f, kVolumeParams.weight_sample * wn * wd);

  float weight = (uchar)fmax(geoemtry_helper.weight_sample * 1.5f *
                     (1.0f - geoemtry_helper.NormalizeDepth(depth,
                                            sensor_params.min_depth_range,
                                            sensor_params.max_depth_range)),
                     1.0f);
  float truncation = geoemtry_helper.truncate_distance(depth);
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

void UpdateBlockArray(EntryArray& candidate_entries,
                      HashTable&  hash_table,
                      BlockArray& blocks,
                      Mesh& mesh,
                      Sensor &sensor,
                      GeometryHelper& geoemtry_helper) {
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
              geoemtry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}