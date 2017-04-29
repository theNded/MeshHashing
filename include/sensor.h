//
// Created by wei on 17-3-20.
//

#ifndef MRF_VH_SENSOR_H
#define MRF_VH_SENSOR_H

#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include "params.h"

/// constant.cu
extern __constant__ SensorParams kSensorParams;
extern void SetConstantSensorParams(const SensorParams& params);

/// At first get rid of CUDARGBDAdaptor and RGBDSensor, use it directly
struct SensorData {
  /// Raw data
  float*		depth_image_;
  float4*		color_image_;

  /// Texture-binded data
  cudaArray*	depth_array_;
  cudaArray*	color_array_;
  cudaChannelFormatDesc depth_channel_desc;
  cudaChannelFormatDesc color_channel_desc;
};

class Sensor {
public:
  Sensor(SensorParams &params);
  ~Sensor();

  int Process(cv::Mat &depth, cv::Mat &color);
  float4* ColorizeDepthImage() const;

  void set_transform(float4x4 w_T_c) {
    w_T_c_ = w_T_c;
    c_T_w_ = w_T_c_.getInverse();
  }
  const float4x4& w_T_c() const { return w_T_c_; }
  const float4x4& c_T_w() const { return c_T_w_; }

  uint width() const { return sensor_params_.width; }
  uint height() const { return sensor_params_.height; }

  const SensorData& sensor_data() { return sensor_data_; }
  const SensorParams& sensor_params() { return sensor_params_; }

private:
  /// sensor data
  SensorData		sensor_data_;
  SensorParams	sensor_params_;

  float4x4      w_T_c_; // camera -> world
  float4x4      c_T_w_;

  //! hsv depth for visualization
  float4* colored_depth_image_;
  short*  depth_image_buffer_;
  uchar*  color_image_buffer_;

  void DepthCPUtoGPU(cv::Mat &depth);
  void ColorCPUtoGPU(cv::Mat &color);
};


#endif //MRF_VH_SENSOR_H
