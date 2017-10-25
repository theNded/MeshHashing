/// 16 threads per block

#include "rgbd_sensor.h"
#include "geometry/geometry_helper.h"
#include "visualization/color_util.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include <glog/logging.h>
#include <driver_types.h>
#include <extern/cuda/helper_cuda.h>
#include "sensor/preprocess.h"


/// Member functions: (CPU code)
Sensor::Sensor(SensorParams &sensor_params) {
  const uint image_size = sensor_params.height * sensor_params.width;

  params_ = sensor_params; // Is it copy constructing?
  checkCudaErrors(cudaMalloc(&data_.depth_buffer, sizeof(short) * image_size));
  checkCudaErrors(cudaMalloc(&data_.color_buffer, sizeof(uchar4) * image_size));
  checkCudaErrors(cudaMalloc(&data_.depth_data, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc(&data_.color_data, sizeof(float4) * image_size));
  data_.depth_channel_desc = cudaCreateChannelDesc<float>();
  checkCudaErrors(cudaMallocArray(&data_.depth_array,
                                  &data_.depth_channel_desc,
                                  params_.width, params_.height));
  data_.color_channel_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaMallocArray(&data_.color_array,
                                  &data_.color_channel_desc,
                                  params_.width, params_.height));
  data_.depth_texture = 0;
  data_.color_texture = 0;

  BindCUDATexture();
  is_allocated_on_gpu_ = true;
}

Sensor::~Sensor() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(data_.depth_buffer));
    checkCudaErrors(cudaFree(data_.color_buffer));
    checkCudaErrors(cudaFree(data_.depth_data));
    checkCudaErrors(cudaFree(data_.color_data));
    checkCudaErrors(cudaFreeArray(data_.depth_array));
    checkCudaErrors(cudaFreeArray(data_.color_array));
  }
}

void Sensor::BindCUDATexture() {
  cudaResourceDesc depth_resource;
  memset(&depth_resource, 0, sizeof(depth_resource));
  depth_resource.resType = cudaResourceTypeArray;
  depth_resource.res.array.array = data_.depth_array;

  cudaTextureDesc depth_tex_desc;
  memset(&depth_tex_desc, 0, sizeof(depth_tex_desc));
  depth_tex_desc.readMode = cudaReadModeElementType;

  if (data_.depth_texture != 0)
    checkCudaErrors(cudaDestroyTextureObject(data_.depth_texture));
  checkCudaErrors(cudaCreateTextureObject(&data_.depth_texture,
                                          &depth_resource,
                                          &depth_tex_desc,
                                          NULL));

  cudaResourceDesc color_resource;
  memset(&color_resource, 0, sizeof(color_resource));
  color_resource.resType = cudaResourceTypeArray;
  color_resource.res.array.array = data_.color_array;

  cudaTextureDesc color_tex_desc;
  memset(&color_tex_desc, 0, sizeof(color_tex_desc));
  color_tex_desc.readMode = cudaReadModeElementType;

  if (data_.color_texture != 0)
    checkCudaErrors(cudaDestroyTextureObject(data_.color_texture));
  checkCudaErrors(cudaCreateTextureObject(&data_.color_texture, &color_resource, &color_tex_desc, NULL));
}

int Sensor::Process(cv::Mat &depth, cv::Mat &color) {
  // TODO(wei): deal with distortion
  /// Disable all filters at current

  ConvertDepthFormat(depth, data_.depth_buffer, data_.depth_data, params_);
  ConvertColorFormat(color, data_.color_buffer, data_.color_data, params_);

  /// Array used as texture in mapper
  checkCudaErrors(cudaMemcpyToArray(data_.depth_array, 0, 0,
                                    data_.depth_data,
                                    sizeof(float)*params_.height*params_.width,
                                    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpyToArray(data_.color_array, 0, 0,
                                    data_.color_data,
                                    sizeof(float4)*params_.height*params_.width,
                                    cudaMemcpyDeviceToDevice));
  BindCUDATexture();
  return 0;
}



