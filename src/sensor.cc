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

CUDARGBDSensor::CUDARGBDSensor() {
  m_bFilterDepthValues = false;
  m_fBilateralFilterSigmaD = 1.0f;
  m_fBilateralFilterSigmaR = 1.0f;

  m_bFilterIntensityValues = false;
  m_fBilateralFilterSigmaDIntensity = 1.0f;
  m_fBilateralFilterSigmaRIntensity = 1.0f;

  d_depthHSV = NULL;
}

CUDARGBDSensor::~CUDARGBDSensor() {
  free();
}

void CUDARGBDSensor::free() {
  checkCudaErrors(cudaFree(d_depthHSV));

  m_depthCameraData.free();
}

int CUDARGBDSensor::alloc(unsigned int width, unsigned int height, DepthCameraParams &params) {

  const unsigned int bufferDimDepth = width * height;
  checkCudaErrors(cudaMalloc(&d_depthHSV, sizeof(float4)*bufferDimDepth));

  /// Parameter settings
  m_depthCameraParams = params; // Is it copy constructing?
  m_depthCameraData.alloc(m_depthCameraParams);

  return 0;
}

int CUDARGBDSensor::process(cv::Mat &depth, cv::Mat &color) {
  /// Distortion is not yet dealt with yet
  /// Disable all filter at current

  /// Input:  CPU short*
  /// Output: GPU float *
  convertDepthRawToFloat(m_depthCameraData.d_depthData, (short *)depth.data,
                         m_depthCameraParams.m_imageWidth, m_depthCameraParams.m_imageHeight,
                         m_depthCameraParams.m_sensorDepthWorldMin,
                         m_depthCameraParams.m_sensorDepthWorldMax);

  /// Input:  CPU uchar4 *
  /// Output: GPU float4 *
  convertColorRawToFloat4(m_depthCameraData.d_colorData, color.data,
                          m_depthCameraParams.m_imageWidth,
                          m_depthCameraParams.m_imageHeight);

  /// TODO: Put intrinsics into DepthCameraParams
  m_depthCameraData.updateParams(getDepthCameraParams());

  /// Array used as texture in mapper
  checkCudaErrors(cudaMemcpyToArray(m_depthCameraData.d_depthArray, 0, 0, m_depthCameraData.d_depthData,
                                    sizeof(float)*m_depthCameraParams.m_imageHeight*m_depthCameraParams.m_imageWidth,
                                    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpyToArray(m_depthCameraData.d_colorArray, 0, 0, m_depthCameraData.d_colorData,
                                    sizeof(float4)*m_depthCameraParams.m_imageHeight*m_depthCameraParams.m_imageWidth,
                                    cudaMemcpyDeviceToDevice));

  return 0;
}

//! enables bilateral filtering of the depth value
void CUDARGBDSensor::setFiterDepthValues(bool b, float sigmaD, float sigmaR) {
  m_bFilterDepthValues = b;
  m_fBilateralFilterSigmaD = sigmaD;
  m_fBilateralFilterSigmaR = sigmaR;
}

void CUDARGBDSensor::setFiterIntensityValues(bool b, float sigmaD, float sigmaR) {
  m_bFilterIntensityValues = b;
  m_fBilateralFilterSigmaDIntensity = sigmaD;
  m_fBilateralFilterSigmaRIntensity = sigmaR;
}

unsigned int CUDARGBDSensor::getDepthWidth() const {
  return m_depthCameraParams.m_imageWidth;
}

unsigned int CUDARGBDSensor::getDepthHeight() const {
  return m_depthCameraParams.m_imageHeight;
}

unsigned int CUDARGBDSensor::getColorWidth() const {
  return m_depthCameraParams.m_imageHeight;
}

unsigned int CUDARGBDSensor::getColorHeight() const {
  return m_depthCameraParams.m_imageWidth;
}

float4* CUDARGBDSensor::getAndComputeDepthHSV() const {
  depthToHSV(d_depthHSV, m_depthCameraData.d_depthData, getDepthWidth(), getDepthHeight(),
             m_depthCameraParams.m_sensorDepthWorldMin,
             m_depthCameraParams.m_sensorDepthWorldMax);
  return d_depthHSV;
}