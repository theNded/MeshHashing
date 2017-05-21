//
// Created by wei on 17-3-20.
//

#ifndef VH_SENSOR_H
#define VH_SENSOR_H

#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include "params.h"

/// At first get rid of CUDARGBDAdaptor and RGBDSensor, use it directly
struct SensorDataGPU {
  /// Raw data
  float*		depth_image;
  float4*		color_image;

  /// Texture-binded data
  cudaArray*	depth_array;
  cudaArray*	color_array;
  cudaChannelFormatDesc depth_channel_desc;
  cudaChannelFormatDesc color_channel_desc;
};

class Sensor {
private:
  /// sensor data
  SensorDataGPU	gpu_data_;
  SensorParams	sensor_params_;
  // mysterious padding for alignment

  float4x4      w_T_c_; // camera -> world
  float4x4      c_T_w_;

  /// sensor data cpu
  float4* colored_depth_image_;
  short*  depth_imagebuffer_;
  uchar*  color_imagebuffer_;

  void DepthCPUtoGPU(cv::Mat &depth);
  void ColorCPUtoGPU(cv::Mat &color);

public:
  Sensor(SensorParams &params);
  ~Sensor();

  void BindGPUTexture();
  int Process(cv::Mat &depth, cv::Mat &color);
  float4* ColorizeDepthImage() const;

  void set_transform(float4x4 w_T_c) {
    w_T_c_ = w_T_c;
    c_T_w_ = w_T_c_.getInverse();
  }
  const float4x4& w_T_c() const {
    return w_T_c_;
  }
  const float4x4& c_T_w() const {
    return c_T_w_;
  }
  uint width() const {
    return sensor_params_.width;
  }
  uint height() const {
    return sensor_params_.height;
  }
  const SensorDataGPU& gpu_data() {
    return gpu_data_;
  }
  const SensorParams& sensor_params() {
    return sensor_params_;
  }
};

#endif //VH_SENSOR_H
