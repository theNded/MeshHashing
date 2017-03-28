//
// Created by wei on 17-3-16.
//

#ifndef MRF_VH_DEPTH_CAMERA_H
#define MRF_VH_DEPTH_CAMERA_H

#include "common.h"
#include "sensor_param.h"

#include <helper_cuda.h>
#include <matrix.h>

/// constant.cu
extern __constant__ DepthCameraParams c_depthCameraParams;
extern void updateConstantDepthCameraParams(const DepthCameraParams& params);

struct DepthCameraData {
  ///////////////
  // Host part //
  ///////////////
  __device__ __host__
  DepthCameraData() {
    d_depthData = NULL;
    d_colorData = NULL;
    d_depthArray = NULL;
    d_colorArray = NULL;
  }

  __host__
  void alloc(const DepthCameraParams& params) { //! todo resizing???
    checkCudaErrors(cudaMalloc(&d_depthData, sizeof(float) * params.m_imageWidth * params.m_imageHeight));
    checkCudaErrors(cudaMalloc(&d_colorData, sizeof(float4) * params.m_imageWidth * params.m_imageHeight));

    h_depthChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&d_depthArray, &h_depthChannelDesc, params.m_imageWidth, params.m_imageHeight));
    h_colorChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&d_colorArray, &h_colorChannelDesc, params.m_imageWidth, params.m_imageHeight));

    /// Added here
    updateParams(params);
  }

  __host__
  void updateParams(const DepthCameraParams& params) {
    updateConstantDepthCameraParams(params);
  }

  __host__
  void free() {
    if (d_depthData) checkCudaErrors(cudaFree(d_depthData));
    if (d_colorData) checkCudaErrors(cudaFree(d_colorData));
    if (d_depthArray) checkCudaErrors(cudaFreeArray(d_depthArray));
    if (d_colorArray) checkCudaErrors(cudaFreeArray(d_colorArray));

    d_depthData = NULL;
    d_colorData = NULL;
    d_depthArray = NULL;
    d_colorArray = NULL;
  }


  /////////////////
  // Device part //
  /////////////////
  static inline const DepthCameraParams& params() {
    return c_depthCameraParams;
  }

  ///////////////////////////////////////////////////////////////
  // Camera to Screen
  ///////////////////////////////////////////////////////////////
  /// R^3 -> R^2
  __device__
  static inline float2 cameraToKinectScreenFloat(const float3& pos)	{
    return make_float2(
            pos.x*c_depthCameraParams.fx/pos.z + c_depthCameraParams.mx,
            pos.y*c_depthCameraParams.fy/pos.z + c_depthCameraParams.my);
  }

  __device__
  static inline int2 cameraToKinectScreenInt(const float3& pos)	{
    float2 pImage = cameraToKinectScreenFloat(pos);
    return make_int2(pImage + make_float2(0.5f, 0.5f));
  }

  __device__
  static inline uint2 cameraToKinectScreen(const float3& pos)	{
    int2 p = cameraToKinectScreenInt(pos);
    return make_uint2(p.x, p.y);
  }

  __device__
  /// R^3 -> [0, 1]^3
  /// maybe used for rendering
  static inline float cameraToKinectProjZ(float z)	{
    return (z - c_depthCameraParams.m_sensorDepthWorldMin)
           /(c_depthCameraParams.m_sensorDepthWorldMax - c_depthCameraParams.m_sensorDepthWorldMin);
  }

  __device__
  static inline float3 cameraToKinectProj(const float3& pos) {
    float2 proj = cameraToKinectScreenFloat(pos);

    float3 pImage = make_float3(proj.x, proj.y, pos.z);

    pImage.x = (2.0f*pImage.x - (c_depthCameraParams.m_imageWidth- 1.0f))/(c_depthCameraParams.m_imageWidth- 1.0f);
    //pImage.y = (2.0f*pImage.y - (c_depthCameraParams.m_imageHeight-1.0f))/(c_depthCameraParams.m_imageHeight-1.0f);
    pImage.y = ((c_depthCameraParams.m_imageHeight-1.0f) - 2.0f*pImage.y)/(c_depthCameraParams.m_imageHeight-1.0f);
    pImage.z = cameraToKinectProjZ(pImage.z);

    return pImage;
  }

  ///////////////////////////////////////////////////////////////
  // Screen to Camera (depth in meters)
  ///////////////////////////////////////////////////////////////
  /// R^2 -> R^3
  __device__
  static inline float3 kinectDepthToSkeleton(uint ux, uint uy, float depth)	{
    const float x = ((float)ux-c_depthCameraParams.mx) / c_depthCameraParams.fx;
    const float y = ((float)uy-c_depthCameraParams.my) / c_depthCameraParams.fy;
    //const float y = (c_depthCameraParams.my-(float)uy) / c_depthCameraParams.fy;
    return make_float3(depth*x, depth*y, depth);
  }

  ///////////////////////////////////////////////////////////////
  // RenderScreen to Camera -- ATTENTION ASSUMES [1,0]-Z range!!!!
  ///////////////////////////////////////////////////////////////
  /// [0, 1]^3 -> R^3
  __device__
  static inline float kinectProjToCameraZ(float z) {
    return z * (c_depthCameraParams.m_sensorDepthWorldMax - c_depthCameraParams.m_sensorDepthWorldMin)
           + c_depthCameraParams.m_sensorDepthWorldMin;
  }

  // z has to be in [0, 1]
  __device__
  static inline float3 kinectProjToCamera(uint ux, uint uy, float z)	{
    // TODO: wei check if its correct
    float fSkeletonZ = kinectProjToCameraZ(z);
    return kinectDepthToSkeleton(ux, uy, fSkeletonZ);
  }

  __device__
  static inline bool isInCameraFrustumApprox(const float4x4& viewMatrixInverse, const float3& pos) {
    float3 pCamera = viewMatrixInverse * pos;
    float3 pProj = cameraToKinectProj(pCamera);
    //pProj *= 1.5f;	//TODO THIS IS A HACK FIX IT :)
    pProj *= 0.95;
    return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f);
  }

  /// Raw data
  float*		d_depthData;	//depth data of the current frame (in screen space):: TODO data allocation lives in RGBD Sensor
  float4*		d_colorData;

  //uchar4*		d_colorData;	//color data of the current frame (in screen space):: TODO data allocation lives in RGBD Sensor

  // cuda arrays for texture access
  /// READ during fusion
  cudaArray*	d_depthArray;
  cudaArray*	d_colorArray;
  cudaChannelFormatDesc h_depthChannelDesc;
  cudaChannelFormatDesc h_colorChannelDesc;
};

#endif //MRF_VH_DEPTH_CAMERA_H
