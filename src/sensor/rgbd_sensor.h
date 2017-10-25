//
// Created by wei on 17-3-20.
//

#ifndef VH_SENSOR_H
#define VH_SENSOR_H

#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include "core/params.h"
#include "sensor/preprocess.h"

/// At first get rid of CUDARGBDAdaptor and Sensor, use it directly
struct SensorData {
  /// sensor data raw
  short*  depth_buffer;
  uchar4* color_buffer;

  /// Reformatted data
  float*		depth_data;
  float4*		color_data;

  /// Texture-binded data
  cudaArray*	depth_array;
  cudaArray*	color_array;

  cudaTextureObject_t depth_texture;
  cudaTextureObject_t color_texture;
  cudaChannelFormatDesc depth_channel_desc;
  cudaChannelFormatDesc color_channel_desc;
};

class Sensor {
public:
  Sensor() = default;
  explicit Sensor(SensorParams &params);
  ~Sensor();
  void BindCUDATexture();

  int Process(cv::Mat &depth, cv::Mat &color);

  void set_transform(float4x4 wTc) {
    wTc_ = wTc;
    cTw_ = wTc_.getInverse();
  }
  const float4x4& w_T_c() const {
    return wTc_;
  }
  const float4x4& c_T_w() const {
    return cTw_;
  }
  uint width() const {
    return params_.width;
  }
  uint height() const {
    return params_.height;
  }
  const SensorData& data() {
    return data_;
  }
  const SensorParams& sensor_params() {
    return params_;
  }

private:
  bool is_allocated_on_gpu_ = false;

  /// sensor data
  SensorData	  data_;
  SensorParams	params_;

  float4x4      wTc_; // camera -> world
  float4x4      cTw_;
};

#endif //VH_SENSOR_H
