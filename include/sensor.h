//
// Created by wei on 17-3-20.
//

#ifndef MRF_VH_SENSOR_H
#define MRF_VH_SENSOR_H

#include <opencv2/opencv.hpp>
#include "sensor_data.h"
#include "sensor_param.h"

/// At first get rid of CUDARGBDAdaptor and RGBDSensor, use it directly
class Sensor {
public:
  Sensor(SensorParams &params);
  ~Sensor();

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

  /// Get data from CUDARGBDAdapter, which reads from RGBDSensor
  int Process(cv::Mat &depth, cv::Mat &color);

  unsigned int getDepthWidth() const;
  unsigned int getDepthHeight() const;
  unsigned int getColorWidth() const;
  unsigned int getColorHeight() const;

  //! the depth camera data (lives on the GPU)
  const SensorData& getSensorData() {
    return sensor_data_;
  }

  //! the depth camera parameter struct (lives on the CPU)
  const SensorParams& getSensorParams() {
    return sensor_params_;
  }

  //! computes and returns the depth map in hsv
  float4* getAndComputeDepthHSV() const;

private:
  /// sensor data
  SensorData		sensor_data_;
  SensorParams	sensor_params_;

  float4x4      w_T_c_; // camera -> world
  float4x4      c_T_w_;

  //! hsv depth for visualization
  float4* depth_image_HSV;
};


#endif //MRF_VH_SENSOR_H
