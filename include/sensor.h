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
  Sensor();
  ~Sensor();

  void Free();
  int alloc(unsigned int width, unsigned int height, SensorParams &params);

  /// Get data from CUDARGBDAdapter, which reads from RGBDSensor
  int process(cv::Mat &depth, cv::Mat &color);

  //! enables bilateral filtering of the depth value
  void setFiterDepthValues(bool b = true, float sigmaD = 1.0f, float sigmaR = 1.0f);
  void setFiterIntensityValues(bool b = true, float sigmaD = 1.0f, float sigmaR = 1.0f);

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

  /// filters the raw data
  /// Check out this later
  bool  m_bFilterDepthValues;
  float m_fBilateralFilterSigmaD;
  float m_fBilateralFilterSigmaR;

  bool  m_bFilterIntensityValues;
  float m_fBilateralFilterSigmaDIntensity;
  float m_fBilateralFilterSigmaRIntensity;

  //! hsv depth for visualization
  float4* d_depthHSV;
};


#endif //MRF_VH_SENSOR_H
