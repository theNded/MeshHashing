#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <extern/cuda/helper_cuda.h>
#include "core/params.h"
#include "visualization/colorize.h"
#include "visualization/color_util.h"

__device__
float3 DepthToRGB(float depth, float depthMin, float depthMax) {
  float normalized_depth = (depth - depthMin)/(depthMax - depthMin);
  float x = 1.0f-normalized_depth;
  if (x < 0.0f)	x = 0.0f;
  if (x > 1.0f)	x = 1.0f;

  x = 360.0f*x - 120.0f;
  if (x < 0.0f) x += 359.0f;
  return HSVToRGB(make_float3(x, 1.0f, 0.5f));
}

__global__
void ColorizeDepthKernel(float4 *dst, float *src,
                         uint width, uint height,
                         float min_depth_range,
                         float max_depth_range) {
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= 0 && x < width && y >= 0 && y < height) {

    float depth = src[y*width + x];
    if (depth != MINF && depth != 0.0f
        && depth >= min_depth_range
        && depth <= max_depth_range) {
      float3 c = DepthToRGB(depth, min_depth_range, max_depth_range);
      dst[y*width + x] = make_float4(c, 1.0f);
    } else {
      dst[y*width + x] = make_float4(0.0f);
    }
  }
}

void ColorizeDepth(float4* colorized_depth, float* depth, SensorParams& params) {
  const uint threads_per_block = 16;
  const dim3 grid_size((params.width + threads_per_block - 1)/threads_per_block,
                       (params.height + threads_per_block - 1)/threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  ColorizeDepthKernel<<<grid_size, block_size>>>(
      colorized_depth, depth,
          params.width, params.height,
          params.min_depth_range,
          params.max_depth_range);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}
