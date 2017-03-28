//
// Created by wei on 17-3-20.
//

#ifndef MRF_VH_SENSOR_H
#define MRF_VH_SENSOR_H

#include <opencv2/opencv.hpp>
#include "sensor_data.h"
#include "sensor_param.h"

/// At first get rid of CUDARGBDAdaptor and RGBDSensor, use it directly
class CUDARGBDSensor {
public:
  CUDARGBDSensor();
  ~CUDARGBDSensor();

  void free();
  int alloc(unsigned int width, unsigned int height, DepthCameraParams &params);

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
  const DepthCameraData& getDepthCameraData() {
    return m_depthCameraData;
  }

  //! the depth camera parameter struct (lives on the CPU)
  const DepthCameraParams& getDepthCameraParams() {
    return m_depthCameraParams;
  }

  //! computes and returns the depth map in hsv
  float4* getAndComputeDepthHSV() const;

private:
  /// sensor data
  DepthCameraData		m_depthCameraData;
  DepthCameraParams	m_depthCameraParams;

  /// filters the raw data
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
