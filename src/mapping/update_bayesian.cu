//
// Created by wei on 17-10-25.
//

#include <device_launch_parameters.h>
#include <util/timer.h>
#include "update_bayesian.h"

#include "core/block_array.h"
#include "mapping/update_simple.h"
#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/spatial_query.h"

// https://github.com/parallel-forall/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu
__device__
void AtomicMax(float * const address, const float value) {
  if (* address >= value)
  {
    return;
  }

  int * const address_as_i = (int *)address;
  int old = * address_as_i, assumed;

  do {
    assumed = old;
    if (__int_as_float(assumed) >= value) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __float_as_int(value));
  } while (assumed != old);
}

__global__
void PredictOutlierRatioKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    HashTable hash_table,
    GeometryHelper geometry_helper) {

  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];
  MeshUnit &this_mesh_unit = blocks[entry.ptr].mesh_units[local_idx];

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
  int image_idx = image_pos.x + image_pos.y * sensor_params.width;

  /// 3. Find correspondent depth observation
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= geometry_helper.sdf_upper_bound)
    return;

  float3 point_cam =
      geometry_helper.ImageReprojectToCamera(image_pos.x, image_pos.y, depth,
                                             sensor_params.fx, sensor_params.fy,
                                             sensor_params.cx, sensor_params.cy);

  for (int i = 0; i < N_VERTEX; ++i) {
    if (this_mesh_unit.vertex_ptrs[i] > 0) {
      Vertex& vtx = mesh.vertex(this_mesh_unit.vertex_ptrs[i]);
      float3 n = vtx.normal; // in world coordinate system
      float3 x = vtx.pos;
      float  r = vtx.radius;

      float cos_alpha = dot(normalize(-point_cam), normalize(cTw * n));

      float3 vec = point_cam - cTw * x;
      float proj_dist = dot(vec, normalize(cTw * n));
      float proj_disk = sqrtf(dot(vec, vec) - squaref(proj_dist));

      float w_dist = expf(- squaref(proj_dist - this_voxel.sdf) * this_voxel.inv_sigma2);
      float w_disk = 0.5f + 0.5f / (1.0f + expf(- fabsf(proj_disk / r)));
      // cos (80) = 0.1736
      float w_angle = (cos_alpha - 0.1736f) / (1.0f - 0.1736f);
      w_angle = fmaxf(w_angle, 0.1f);

      float inlier_ratio = w_disk * w_dist * w_angle;
      inlier_ratio = fmaxf(inlier_ratio, 0.1f);
      AtomicMax(&sensor_data.inlier_ratio[image_idx], inlier_ratio);
    }
  }
}

__global__
void UpdateBlocksBayesianKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    HashTable hash_table,
    GeometryHelper geometry_helper) {

  //TODO check if we should load this in shared memory (entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

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
  int image_idx = image_pos.x + image_pos.y * sensor_params.width;

  float x = depth - camera_pos.z;
  float rho = sensor_data.inlier_ratio[image_idx];
  float truncation = geometry_helper.truncate_distance(depth);
  if (x <= -truncation)
    return;
  if (x >= 0.0f) {
    x = fminf(truncation, x);
  } else {
    x = fmaxf(-truncation, x);
  }

//  // Depth filter
  float tau = (depth - 0.4f) * 0.012f + 0.019f;
  // uninitialized
  if (this_voxel.inv_sigma2 == 0) {
    this_voxel.sdf = x;
    this_voxel.inv_sigma2 = 1.0f / squaref(tau);
    this_voxel.a = 0;
    this_voxel.b = 10;
  } else {
    float mu = this_voxel.sdf;
    float squared_sigma = 1.0f / this_voxel.inv_sigma2;
    float squared_tau = squaref(tau);
    float squared_s = 1.0f / (1.0f / squared_sigma + 1.0f / squared_tau);
    float m = squared_s * (mu / squared_sigma + x / squared_tau);

    float C1 = rho * gaussian(x, mu, squared_sigma + squared_tau);
    float d_min = sensor_params.min_depth_range;
    float d_max = sensor_params.max_depth_range;
    float C2 = (depth < d_max && depth >= d_min) ?
               (1-rho) * 1.0f / (5.0f - 0.1f) : 1.0f;
    float sum_C1_C2 = C1 + C2;
    C1 /= sum_C1_C2;
    C2 /= sum_C1_C2;

    float a = this_voxel.a;
    float b = this_voxel.b;
    float f = C1*(a+1)/(a+b+1) + C2*a/(a+b+1);
    float e = C1*(a+1)*(a+2)/((a+b+1)*(a+b+2)) + C2*a*(a+1)/((a+b+1)*(a+b+2));

    this_voxel.sdf = C1 * m + C2 * mu;
    this_voxel.inv_sigma2 = 1.0f / (C1 * (squared_s + squaref(m))
                                + C2 * (squared_sigma + squaref(mu))
                                - squaref(this_voxel.sdf));
    this_voxel.a = (e-f) / (f-e/f);
    this_voxel.b = this_voxel.a*(1.0f-f)/f;
  }
}

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
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[local_idx];

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
    if (mesh_unit.vertex_ptrs[i] > 0) {
      Vertex vtx = mesh.vertex(mesh_unit.vertex_ptrs[i]);
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

float PredictOutlierRatio(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    Mesh& mesh,
    Sensor& sensor,
    HashTable& hash_table,
    GeometryHelper& geometry_helper) {
  const uint threads_per_block = BLOCK_SIZE;

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return -1;

  Timer timer;
  timer.Tick();
  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);
  PredictOutlierRatioKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          mesh,
          sensor.data(),
          sensor.sensor_params(),
          sensor.cTw(),
          hash_table,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return timer.Tock();
}


float UpdateBlocksBayesian(
  EntryArray &candidate_entries,
  BlockArray &blocks,
  Sensor &sensor,
  HashTable &hash_table,
  GeometryHelper &geometry_helper
) {
  const uint threads_per_block = BLOCK_SIZE;

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return - 1;

  Timer timer;
  timer.Tick();
  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);
  UpdateBlocksBayesianKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          sensor.data(),
          sensor.sensor_params(),
          sensor.cTw(),
          hash_table,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  return timer.Tock();
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

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return;

  const dim3 grid_size(candidate_entry_count, 1);
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