//
// Created by wei on 17-10-25.
//

#include <device_launch_parameters.h>
#include "linear_equations.h"


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
  input_pos = wTc * input_pos;
  float3 predict_pos = (float3x3::getIdentity() + linear_equations.A[idx]).getInverse()
                       * (input_pos + linear_equations.b[idx]);
  //float3x3 A = linear_equations.A[idx];
  //float3 b = linear_equations.b[idx];
  // TODO: low support -> outlier
  float diff = length(input_pos - predict_pos);
  linear_equations.b[idx] = wTc.getInverse() * predict_pos;
//  if (diff < 1e-6) return;
//  printf("(%d %d) -> \n %f %f %f, %f\n %f %f %f, %f\n %f %f %f, %f\n"
//             "-diff:%f, (%f %f %f) -> (%f %f %f)\n",
//         x, y,
//         A.entries[0],A.entries[1],A.entries[2], b.x,
//         A.entries[3],A.entries[4],A.entries[5], b.y,
//         A.entries[6],A.entries[7],A.entries[8], b.z,
//         diff,
//         input_pos.x, input_pos.y, input_pos.z,
//         predict_pos.x, predict_pos.y, predict_pos.z);
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

void SensorLinearEquations::Alloc(int width, int height) {
  width_ = width;
  height_ = height;

  if (!is_allocated_on_gpu_) {
    cudaMalloc(&A, sizeof(float3x3) * width*height);
    cudaMalloc(&b, sizeof(float3) * width*height);
    is_allocated_on_gpu_ = true;
  }
}

void SensorLinearEquations::Free() {
  if (is_allocated_on_gpu_) {
    cudaFree(A);
    cudaFree(b);
    is_allocated_on_gpu_ = false;
  }
}

void SensorLinearEquations::Reset() {
  if (is_allocated_on_gpu_) {
    cudaMemset(A, 0, sizeof(float3x3)*width_*height_);
    cudaMemset(b, 0, sizeof(float3)*width_*height_);
  }
}