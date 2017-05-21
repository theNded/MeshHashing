/// 16 threads per block

#include "sensor.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include <glog/logging.h>

#define MINF __int_as_float(0xff800000)

////////////////////
/// Texture
////////////////////
texture<float, cudaTextureType2D, cudaReadModeElementType> depth_texture;
texture<float4, cudaTextureType2D, cudaReadModeElementType> color_texture;
void Sensor::BindGPUTexture() {
  checkCudaErrors(cudaBindTextureToArray(depth_texture,
                                         gpu_data_.depth_array,
                                         gpu_data_.depth_channel_desc));
  checkCudaErrors(cudaBindTextureToArray(color_texture,
                                         gpu_data_.color_array,
                                         gpu_data_.color_channel_desc));
  depth_texture.filterMode = cudaFilterModePoint;
  color_texture.filterMode = cudaFilterModePoint;
}

////////////////////
/// Device code
////////////////////
__global__
void DepthCPUtoGPUKernel(float *dst, short *src,
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
void ColorCPUtoGPUKernel(float4* dst, uchar *src,
                         uint width, uint height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  const int idx = y * width + x;

  uchar4 c = make_uchar4(src[4 * idx + 0], src[4 * idx + 1],
                         src[4 * idx + 2], src[4 * idx + 3]);
  bool is_valid = (c.x != 0 && c.y != 0 && c.z != 0);
  dst[idx] = is_valid ? make_float4(c.x / 255.0f, c.y / 255.0f,
                                    c.z / 255.0f, c.w / 255.0f)
                           : make_float4(MINF, MINF, MINF, MINF);
}

/// Util: Depth to RGB
__device__
float3 HSVToRGB(const float3& hsv) {
  float H = hsv.x;
  float S = hsv.y;
  float V = hsv.z;

  float hd = H/60.0f;
  uint h = (uint)hd;
  float f = hd-h;

  float p = V*(1.0f-S);
  float q = V*(1.0f-S*f);
  float t = V*(1.0f-S*(1.0f-f));

  if(h == 0 || h == 6) {
    return make_float3(V, t, p);
  }
  else if(h == 1) {
    return make_float3(q, V, p);
  }
  else if(h == 2) {
    return make_float3(p, V, t);
  }
  else if(h == 3) {
    return make_float3(p, q, V);
  }
  else if(h == 4) {
    return make_float3(t, p, V);
  } else {
    return make_float3(V, p, q);
  }
}

__device__
float3 DepthToRGB(float depth, float depthMin, float depthMax) {
  float depthZeroOne = (depth - depthMin)/(depthMax - depthMin);
  float x = 1.0f-depthZeroOne;
  if (x < 0.0f)	x = 0.0f;
  if (x > 1.0f)	x = 1.0f;

  x = 360.0f*x - 120.0f;
  if (x < 0.0f) x += 359.0f;
  return HSVToRGB(make_float3(x, 1.0f, 0.5f));
}

__global__
void DepthToRGBKernel(float4* dst, float* src,
                      uint width, uint height,
                      float min_depth_range, float max_depth_range) {
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

/// Member functions: (CPU code)
Sensor::Sensor(SensorParams &sensor_params) {
  colored_depth_image_ = NULL;
  const uint image_size = sensor_params.height * sensor_params.width;

  checkCudaErrors(cudaMalloc(&colored_depth_image_, sizeof(float4) * image_size));
  checkCudaErrors(cudaMalloc(&depth_imagebuffer_, sizeof(short) * image_size));
  checkCudaErrors(cudaMalloc(&color_imagebuffer_, 4 * sizeof(uchar) * image_size));

  /// Parameter settings
  sensor_params_ = sensor_params; // Is it copy constructing?
  checkCudaErrors(cudaMalloc(&gpu_data_.depth_image, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc(&gpu_data_.color_image, sizeof(float4) * image_size));

  gpu_data_.depth_channel_desc
          = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  checkCudaErrors(cudaMallocArray(&gpu_data_.depth_array,
                                  &gpu_data_.depth_channel_desc,
                                  sensor_params_.width, sensor_params_.height));
  gpu_data_.color_channel_desc
          = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  checkCudaErrors(cudaMallocArray(&gpu_data_.color_array,
                                  &gpu_data_.color_channel_desc,
                                  sensor_params_.width, sensor_params_.height));
}

Sensor::~Sensor() {
  checkCudaErrors(cudaFree(colored_depth_image_));
  checkCudaErrors(cudaFree(depth_imagebuffer_));
  checkCudaErrors(cudaFree(color_imagebuffer_));

  checkCudaErrors(cudaFree(gpu_data_.depth_image));
  checkCudaErrors(cudaFree(gpu_data_.color_image));
  checkCudaErrors(cudaFreeArray(gpu_data_.depth_array));
  checkCudaErrors(cudaFreeArray(gpu_data_.color_array));
}

int Sensor::Process(cv::Mat &depth, cv::Mat &color) {
  // TODO(wei): deal with distortion
  /// Disable all filters at current

  /// Input:  CPU short*
  /// Output: GPU float *
  DepthCPUtoGPU(depth);

  /// Input:  CPU uchar4 *
  /// Output: GPU float4 *
  ColorCPUtoGPU(color);

  /// Array used as texture in mapper
  checkCudaErrors(cudaMemcpyToArray(gpu_data_.depth_array, 0, 0, gpu_data_.depth_image,
                                    sizeof(float)*sensor_params_.height*sensor_params_.width,
                                    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpyToArray(gpu_data_.color_array, 0, 0, gpu_data_.color_image,
                                    sizeof(float4)*sensor_params_.height*sensor_params_.width,
                                    cudaMemcpyDeviceToDevice));
  return 0;
}

//////////
/// Member function: (CPU calling GPU kernels)
void Sensor::DepthCPUtoGPU(cv::Mat& depth) {
  /// First copy cpu data in to cuda short
  uint width = sensor_params_.width;
  uint height = sensor_params_.height;
  uint image_size = width * height;

  checkCudaErrors(cudaMemcpy(depth_imagebuffer_, (short *)depth.data,
                             sizeof(short) * image_size, cudaMemcpyHostToDevice));

  const uint threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1)/threads_per_block,
                       (height + threads_per_block - 1)/threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);
  DepthCPUtoGPUKernel<<<grid_size, block_size>>>(
          gpu_data_.depth_image, depth_imagebuffer_,
          width, height,
          sensor_params_.range_factor,
          sensor_params_.min_depth_range,
          sensor_params_.max_depth_range);
}

__host__
void Sensor::ColorCPUtoGPU(cv::Mat &color) {
  uint width = sensor_params_.width;
  uint height = sensor_params_.height;
  uint image_size = width * height;

  checkCudaErrors(cudaMemcpy(color_imagebuffer_, color.data,
                             4 * sizeof(uchar) * image_size,
                             cudaMemcpyHostToDevice));

  const int threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1)/threads_per_block,
                       (height + threads_per_block - 1)/threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  ColorCPUtoGPUKernel<<<grid_size, block_size>>>(
          gpu_data_.color_image, color_imagebuffer_, width, height);
}

float4* Sensor::ColorizeDepthImage() const {
  const uint threads_per_block = 16;
  const dim3 grid_size((sensor_params_.width + threads_per_block - 1)/threads_per_block,
                       (sensor_params_.height + threads_per_block - 1)/threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  DepthToRGBKernel<<<grid_size, block_size>>>(
          colored_depth_image_, gpu_data_.depth_image,
          sensor_params_.width, sensor_params_.height,
          sensor_params_.min_depth_range,
          sensor_params_.max_depth_range);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return colored_depth_image_;
}

