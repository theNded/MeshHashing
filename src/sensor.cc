//
// Created by wei on 17-3-20.
//

#include "sensor.h"
#include "sensor_data.h"
#include <algorithm>
#include <glog/logging.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <matrix.h>

/// extern in cuda
void convertDepthRawToFloat(float* d_output, short* d_input,
                            unsigned int width, unsigned int height,
                            float minDepth, float maxDepth);
void convertColorRawToFloat4(float4* d_output, unsigned char* d_input,
                             unsigned int width, unsigned int height);
void depthToHSV(float4* d_output, float* d_input,
                unsigned int width, unsigned int height,
                float minDepth, float maxDepth);

Sensor::Sensor() {
  m_bFilterDepthValues = false;
  m_fBilateralFilterSigmaD = 1.0f;
  m_fBilateralFilterSigmaR = 1.0f;

  m_bFilterIntensityValues = false;
  m_fBilateralFilterSigmaDIntensity = 1.0f;
  m_fBilateralFilterSigmaRIntensity = 1.0f;

  d_depthHSV = NULL;
}

Sensor::~Sensor() {
  free();
}

void Sensor::free() {
  checkCudaErrors(cudaFree(d_depthHSV));

  sensor_data_.free();
}

int Sensor::alloc(unsigned int width, unsigned int height, SensorParams &params) {

  const unsigned int bufferDimDepth = width * height;
  checkCudaErrors(cudaMalloc(&d_depthHSV, sizeof(float4)*bufferDimDepth));

  /// Parameter settings
  sensor_params_ = params; // Is it copy constructing?
  sensor_data_.alloc(sensor_params_);

  return 0;
}

int Sensor::process(cv::Mat &depth, cv::Mat &color) {
  /// Distortion is not yet dealt with yet
  /// Disable all filter at current

  /// Input:  CPU short*
  /// Output: GPU float *
  convertDepthRawToFloat(sensor_data_.d_depthData, (short *)depth.data,
                         sensor_params_.width, sensor_params_.height,
                         sensor_params_.min_depth_range,
                         sensor_params_.max_depth_range);

  /// Input:  CPU uchar4 *
  /// Output: GPU float4 *
  convertColorRawToFloat4(sensor_data_.d_colorData, color.data,
                          sensor_params_.width,
                          sensor_params_.height);

  /// TODO: Put intrinsics into SensorParams
  sensor_data_.updateParams(getSensorParams());

  /// Array used as texture in mapper
  checkCudaErrors(cudaMemcpyToArray(sensor_data_.d_depthArray, 0, 0, sensor_data_.d_depthData,
                                    sizeof(float)*sensor_params_.height*sensor_params_.width,
                                    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpyToArray(sensor_data_.d_colorArray, 0, 0, sensor_data_.d_colorData,
                                    sizeof(float4)*sensor_params_.height*sensor_params_.width,
                                    cudaMemcpyDeviceToDevice));

  return 0;
}

//! enables bilateral filtering of the depth value
void Sensor::setFiterDepthValues(bool b, float sigmaD, float sigmaR) {
  m_bFilterDepthValues = b;
  m_fBilateralFilterSigmaD = sigmaD;
  m_fBilateralFilterSigmaR = sigmaR;
}

void Sensor::setFiterIntensityValues(bool b, float sigmaD, float sigmaR) {
  m_bFilterIntensityValues = b;
  m_fBilateralFilterSigmaDIntensity = sigmaD;
  m_fBilateralFilterSigmaRIntensity = sigmaR;
}

unsigned int Sensor::getDepthWidth() const {
  return sensor_params_.width;
}

unsigned int Sensor::getDepthHeight() const {
  return sensor_params_.height;
}

unsigned int Sensor::getColorWidth() const {
  return sensor_params_.height;
}

unsigned int Sensor::getColorHeight() const {
  return sensor_params_.width;
}

float4* Sensor::getAndComputeDepthHSV() const {
  depthToHSV(d_depthHSV, sensor_data_.d_depthData, getDepthWidth(), getDepthHeight(),
             sensor_params_.min_depth_range,
             sensor_params_.max_depth_range);
  return d_depthHSV;
}