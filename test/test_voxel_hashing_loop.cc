//
// Created by wei on 17-3-26.
//

#include <string>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <helper_cuda.h>
#include <chrono>

#include <string>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <sensor.h>
#include <ray_caster.h>

#include "mapper.h"
#include "renderer.h"

#include "config_reader.h"

#define ICL_
#ifdef ICL
const std::string kDefaultDatasetPath = "/home/wei/data/ICL/kt2/";
#else
const std::string kDefaultDatasetPath = "/home/wei/data/TUM/rgbd_dataset_freiburg2_xyz/";
#endif

void checkCudaFloatMemory(float *cuda_memory) {
  static float cpu_memory[640 * 480];
  memset(cpu_memory, 0, sizeof(cpu_memory));

  cv::Mat a = cv::Mat(480, 640, CV_32F, cpu_memory);
  checkCudaErrors(cudaMemcpy(cpu_memory, cuda_memory, sizeof(float) * 640 * 480, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 480; ++i)
    for (int j = 0; j < 640; ++j) {
      //LOG(INFO) << a.at<float>(i, j);
    }
  cv::imshow("check", a);
}

cv::Mat checkCudaFloat4Memory(float4 *cuda_memory) {
  static float cpu_memory[640 * 480 * 4];
  cv::Mat a = cv::Mat(480, 640, CV_32FC4, cpu_memory);

  checkCudaErrors(cudaMemcpy(cpu_memory, cuda_memory, sizeof(float) * 4 * 640 * 480, cudaMemcpyDeviceToHost));
#define VIDEO
#ifdef VIDEO
  cv::Mat b = cv::Mat(480, 640, CV_8UC3);
  for (int i = 0; i < 480; ++i) {
    for (int j = 0; j < 640; ++j) {
      cv::Vec4f cf = a.at<cv::Vec4f>(i, j);
      if (std::isinf(cf[0])) {
        b.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
      } else {
        b.at<cv::Vec3b>(i, j) = cv::Vec3b(255 * fabs(cf[0]), 255 * fabs(cf[1]), 255 * fabs(cf[2]));
      }
    }
  }
  cv::imshow("check", b);
#else
  cv::imshow("check", a);
#endif
  cv::waitKey(10);
  return a;
}

void SetConstantSDFParams(const SDFParams& params);

int main() {
  /// Load images
  std::vector<std::string> depth_img_list;
  std::vector<std::string> color_img_list;
  std::vector<float4x4>    wTc;

#ifdef ICL
  LoadICLImageList(kDefaultDatasetPath, depth_img_list, color_img_list);
  LoadICLTrajectory(kDefaultDatasetPath, wTc);
#else
  LoadTUMImageList(kDefaultDatasetPath, depth_img_list, color_img_list);
  LoadTUMTrajectory(kDefaultDatasetPath, wTc);
#endif

  SDFParams sdf_params;
  LoadSDFParams("../test/sdf_params.yml", sdf_params);
  SetConstantSDFParams(sdf_params);

  HashParams hash_params;
  LoadHashParams("../test/hash_params.yml", hash_params);

  /// Sensor
  SensorParams sensor_params;
#ifdef ICL
  sensor_params.fx = 481.2;
  sensor_params.fy = -480;
  sensor_params.cx = 319.5;
  sensor_params.cy = 239.5;
#else
  sensor_params.fx = 517.3;
  sensor_params.fy = 516.5;
  sensor_params.cx = 318.6;
  sensor_params.cy = 255.3;
#endif
  sensor_params.min_depth_range = 0.5f;
  sensor_params.max_depth_range = 5.0f;
  sensor_params.height = 480;
  sensor_params.width = 640;

  RayCasterParams ray_cast_params;
  ray_cast_params.fx = sensor_params.fx;
  ray_cast_params.fy = sensor_params.fy;
  ray_cast_params.cx = sensor_params.cx;
  ray_cast_params.cy = sensor_params.cy;
  ray_cast_params.width = 640;
  ray_cast_params.height = 480;
  ray_cast_params.min_raycast_depth = 0.5f;
  ray_cast_params.max_raycast_depth = 5.5f;
  ray_cast_params.raycast_step = 0.8f * 0.02f;
  ray_cast_params.sample_sdf_threshold = 50.5f * ray_cast_params.raycast_step;
  ray_cast_params.sdf_threshold = 50.0f * ray_cast_params.raycast_step;


  Map voxel_map(hash_params);
  voxel_map.sensor_params() = sensor_params;
  Mapper mapper;

  RayCaster ray_caster(ray_cast_params);
  Sensor sensor(sensor_params);
  sensor.BindSensorDataToTexture();

  /// Process
  float4 *cuda_hsv;
  checkCudaErrors(cudaMalloc(&cuda_hsv, sizeof(float4) * 640 * 480));

  LOG(INFO) << sizeof(SensorParams);
  LOG(INFO) << sizeof(RayCasterParams);
  LOG(INFO) << sizeof(HashParams);
  LOG(INFO) << sizeof(SDFParams);


  //cv::VideoWriter writer("icl-vh.avi", CV_FOURCC('X','V','I','D'), 30, cv::Size(640, 480));
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int frames = 880;
  for (int i = 0; i < frames; ++i) {
    LOG(INFO) << i;
    cv::Mat depth = cv::imread(depth_img_list[i], -1);
    //cv::flip(depth, depth, 2);
    cv::Mat color = cv::imread(color_img_list[i]);
    //cv::flip(color, color, 2);
    cv::cvtColor(color, color, CV_BGR2BGRA);

    sensor.Process(depth, color);
    float4x4 T = wTc[0].getInverse() * wTc[i];
    sensor.set_transform(T);
    //T = T.getInverse();

    mapper.Integrate(&voxel_map, &sensor, NULL);

    ray_caster.Cast(&voxel_map, T.getInverse());

    checkCudaFloat4Memory(ray_caster.ray_caster_data().normal_image);

    cv::waitKey(1);
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  LOG(INFO) << "Total time: " << seconds.count();
  LOG(INFO) << "Fps: " << frames / seconds.count();

  checkCudaErrors(cudaFree(cuda_hsv));
  voxel_map.Debug();
}