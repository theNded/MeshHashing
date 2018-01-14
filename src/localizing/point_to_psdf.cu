//
// Created by wei on 17-12-23.
//

#include <device_launch_parameters.h>
#include "point_to_psdf.h"
#include "geometry/spatial_query.h"

template<unsigned int N, unsigned int M>
__device__
void AtomicAdd(matNxM<N, M> *A, const matNxM<N, M> dA) {
  const int n = N * M;
#pragma unroll 1
  for (int i = 0; i < n; ++i) {
    atomicAdd(&(A->entries[i]), dA.entries[i]);
  }
}

__global__
void PointToSurfaceKernel(
    BlockArray blocks,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 wTc,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    mat6x6 *A,
    mat6x1 *b,
    float *err,
    int *count
) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= sensor_params.width || y >= sensor_params.height) {
    return;
  }
  if (x % 2 == 0 || y % 2 == 0)
    return;

  float depth = tex2D<float>(sensor_data.depth_texture, x, y);
  if (depth == MINF || depth == 0.0f
      || depth >= geometry_helper.sdf_upper_bound)
    return;

  float3 point_cam = geometry_helper.ImageReprojectToCamera(x,
                                                            y,
                                                            depth,
                                                            sensor_params.fx,
                                                            sensor_params.fy,
                                                            sensor_params.cx,
                                                            sensor_params.cy);
  float3 point_world = wTc * point_cam;
  Voxel voxel;
  bool valid = GetSpatialValue(point_world, blocks, hash_table,
                               geometry_helper, &voxel);
  if (!valid) return;
  float d = voxel.sdf;

  float3 grad;
  valid = GetSpatialSDFGradient(point_world, blocks, hash_table,
                                geometry_helper, &grad);
  if (! valid) return;

  // A = \sum
  // \nabla D * [- y_x | I]
  // [dDx, dDy, dDz] * [ 1 0 0 | 0 z -y ]
  //                   [ 0 1 0 | -z 0 x ]
  //                   [ 0 0 1 | y -x 0 ]
  float dDdx1_data[6] = {
      grad.x, grad.y, grad.x,
      (-grad.y * point_world.z + grad.z * point_world.y),
      (-grad.z * point_world.x + grad.x * point_world.z),
      (-grad.x * point_world.y + grad.y * point_world.x)
  };
  mat6x1 dDdxi(dDdx1_data);
  mat6x6 dA = dDdxi * dDdxi.getTranspose();
  AtomicAdd<6, 6>(A, dA);
  AtomicAdd<6, 1>(b, d * dDdxi);
  atomicAdd(err, d * d);
  //printf("%d %d -> %f\n", x, y, d);
  atomicAdd(count, 1);
}

float PointToSurface(BlockArray &blocks,
                     Sensor &sensor,
                     HashTable &hash_table,
                     GeometryHelper &geometry_helper,
                     mat6x6 &cpu_A,
                     mat6x1 &cpu_b,
                     int &cpu_count) {
  const uint threads_per_block = 16;

  const dim3
      grid_size((sensor.width() + threads_per_block - 1) / threads_per_block,
                (sensor.height() + threads_per_block - 1) / threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  mat6x6 *A;
  mat6x1 *b;
  float *err, cpu_err = 0;
  int *count;
  cpu_A.setZero();
  cpu_b.setZero();
  cpu_count = 0;

  checkCudaErrors(cudaMalloc(&A, sizeof(mat6x6)));
  checkCudaErrors(cudaMemcpy(A, &cpu_A, sizeof(mat6x6),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&b, sizeof(mat6x1)));
  checkCudaErrors(cudaMemcpy(b, &cpu_b, sizeof(mat6x1),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&count, sizeof(int)));
  checkCudaErrors(cudaMemcpy(count, &cpu_count, sizeof(int),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&err, sizeof(float)));
  checkCudaErrors(cudaMemcpy(err, &cpu_err, sizeof(float),
                             cudaMemcpyHostToDevice));

  PointToSurfaceKernel << < grid_size, block_size >> > (
      blocks,
          sensor.data(),
          sensor.sensor_params(),
          sensor.wTc(),
          hash_table,
          geometry_helper,
          A,
          b,
          err,
          count
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&cpu_A, A, sizeof(mat6x6),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&cpu_b, b, sizeof(mat6x1),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&cpu_count, count, sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&cpu_err, err, sizeof(float),
                             cudaMemcpyDeviceToHost));

  return cpu_err;
}
