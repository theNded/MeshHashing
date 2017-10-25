#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <extern/cuda/helper_cuda.h>
#include "core/params.h"
#include "preprocess.h"

#define MINF __int_as_float(0xff800000)

__global__
void ConvertDepthFormatKernel(float *dst, short *src,
                         uint width, uint height,
                         float range_factor,
                         float min_depth_range,
                         float max_depth_range) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  const int idx = y * width + x;
  /// Convert mm -> m
  const float depth = range_factor * src[idx];
  bool is_valid = (depth >= min_depth_range && depth <= max_depth_range);
  dst[idx] = is_valid ? depth : MINF;
}

__global__
void ConvertColorFormatKernel(float4 *dst, uchar4 *src,
                              uint width, uint height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  const int idx = y * width + x;

  uchar4 c = src[idx];
  bool is_valid = (c.x != 0 && c.y != 0 && c.z != 0);
  dst[idx] = is_valid ? make_float4(c.x / 255.0f, c.y / 255.0f,
                                    c.z / 255.0f, c.w / 255.0f)
                      : make_float4(MINF, MINF, MINF, MINF);
}

//////////
/// Member function: (CPU calling GPU kernels)
void ConvertDepthFormat(cv::Mat& depth_img,
                        short* depth_buffer,
                        float* depth_data,
                        SensorParams& params) {
  /// First copy cpu data in to cuda short
  uint width = params.width;
  uint height = params.height;
  uint image_size = width * height;

  checkCudaErrors(cudaMemcpy(depth_buffer, (short *)depth_img.data,
                             sizeof(short) * image_size,
                             cudaMemcpyHostToDevice));

  const uint threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1)/threads_per_block,
                       (height + threads_per_block - 1)/threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);
  ConvertDepthFormatKernel<<<grid_size, block_size>>>(
      depth_data,
          depth_buffer,
          width, height,
          params.range_factor,
          params.min_depth_range,
          params.max_depth_range);
}

__host__
void ConvertColorFormat(cv::Mat &color_img,
                        uchar4* color_buffer,
                        float4* color_data,
                        SensorParams& params) {

  uint width = params.width;
  uint height = params.height;
  uint image_size = width * height;

  checkCudaErrors(cudaMemcpy(color_buffer, color_img.data,
                             sizeof(uchar4) * image_size,
                             cudaMemcpyHostToDevice));

  const int threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1)/threads_per_block,
                       (height + threads_per_block - 1)/threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  ConvertColorFormatKernel <<<grid_size, block_size>>>(
      color_data,
          color_buffer,
          width,
          height);
}
