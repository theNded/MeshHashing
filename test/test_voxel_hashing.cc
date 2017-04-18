//
// Created by wei on 17-3-20.
//

#include <string>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <helper_cuda.h>
#include <sensor.h>
#include <ray_caster.h>

#include "mapper.h"
#include "sensor_data.h"
#include "renderer.h"

const std::string kDefaultDatasetPath = "/home/wei/data/TUM/rgbd_dataset_freiburg1_xyz/";

void DepthToRGBCudaHost(float4* d_output, float* d_input,
                unsigned int width, unsigned int height,
                float minDepth, float maxDepth);

void LoadImageList(std::string dataset_path, std::string dataset_txt,
                   std::vector<std::string> &image_name_list) {
  std::ifstream list_stream(dataset_path + dataset_txt);
  std::string file_name;

  std::getline(list_stream, file_name);
  std::getline(list_stream, file_name);
  std::getline(list_stream, file_name);

  std::string time_stamp;
  while (list_stream >> time_stamp >> file_name) {
    image_name_list.push_back(dataset_path + "/" + file_name);
  }
}

void LoadTrajectory(std::string dataset_path, std::vector<float4x4> &wTc_list) {
  std::ifstream list_stream(dataset_path + "depth_gt_associations.txt");
  std::string ts_img, img_path, ts_gt;
  float tx, ty, tz, qx, qy, qz, qw;
  while (list_stream >> ts_img >> img_path >> ts_gt
                     >> tx >> ty >> tz
                     >> qx >> qy >> qz >> qw) {
    float4x4 wTc;
    wTc.setIdentity();

    wTc.m11 = 1 - 2 * qy * qy - 2 * qz * qz;
    wTc.m12 = 2 * qx * qy - 2 * qz * qw;
    wTc.m13 = 2 * qx * qz + 2 * qy * qw;
    wTc.m14 = tx;
    wTc.m21 = 2 * qx * qy + 2 * qz * qw;
    wTc.m22 = 1 - 2 * qx * qx - 2 * qz * qz;
    wTc.m23 = 2 * qy * qz - 2 * qx * qw;
    wTc.m24 = ty;
    wTc.m31 = 2 * qx * qz - 2 * qy * qw;
    wTc.m32 = 2 * qy * qz + 2 * qx * qw;
    wTc.m33 = 1 - 2 * qx * qx - 2 * qy * qy;
    wTc.m34 = tz;
    wTc.m44 = 1;
    wTc_list.push_back(wTc);
  }
}

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
  cv::waitKey(-1);
}

void checkCudaFloat4Memory(float4 *cuda_memory) {
  static float cpu_memory[640 * 480 * 4];
  cv::Mat a = cv::Mat(480, 640, CV_32FC4, cpu_memory);

  checkCudaErrors(cudaMemcpy(cpu_memory, cuda_memory, sizeof(float) * 4 * 640 * 480, cudaMemcpyDeviceToHost));

  cv::imshow("check", a);
  cv::waitKey(-1);
}

int main() {
  /// Load images
  std::vector<std::string> depth_img_list;
  std::vector<std::string> color_img_list;
  std::vector<float4x4>    wTc;

  LoadImageList(kDefaultDatasetPath, "depth.txt", depth_img_list);
  LoadImageList(kDefaultDatasetPath, "rgb.txt", color_img_list);
  LoadTrajectory(kDefaultDatasetPath, wTc);

  /// Mapper
  HashParams hash_params;
  hash_params.bucket_count = 500000;
  hash_params.bucket_size = 10;
  hash_params.linked_list_size = 7;
  hash_params.block_count = 1000000;
  hash_params.block_size = 8;
  hash_params.voxel_size = 0.004;

  hash_params.sdf_upper_bound = 4.0;
  hash_params.truncation_distance_scale = 0.01;
  hash_params.truncation_distance = 0.02;
  hash_params.weight_sample = 10;
  hash_params.weight_upper_bound = 255;

  Mapper mapper(hash_params);
  //mapper.debugHash();
  /// Only to alloc cuda memory, suppose its ok

  /// Sensor
  SensorParams sensor_params;
  sensor_params.fx = 517.306408f;
  sensor_params.fy = 516.469215f;
  sensor_params.cx = 318.643040f;
  sensor_params.cy = 255.313989f;

  sensor_params.min_depth_range = 0.5f;
  sensor_params.max_depth_range = 5.0f;
  sensor_params.height = 480;
  sensor_params.width = 640;

  Sensor sensor;
  sensor.alloc(640, 480, sensor_params);
  /// suppose its ok
  /// check camera params

  float4x4 T; T.setIdentity();
  float4x4 K; K.setIdentity();
  K.m11 = sensor_params.fx;
  K.m13 = sensor_params.cx;
  K.m22 = sensor_params.fy;
  K.m23 = sensor_params.cy;

  /// Ray Caster
  RayCasterParams ray_cast_params;
  ray_cast_params.c_T_w_ = T;
  ray_cast_params.w_T_c_ = T.getInverse();
  ray_cast_params.m_intrinsics = K;
  ray_cast_params.m_intrinsicsInverse = K.getInverse();
  ray_cast_params.m_width = 640;
  ray_cast_params.m_height = 480;
  ray_cast_params.m_minDepth = 0.5f;
  ray_cast_params.m_maxDepth = 5.0f;
  ray_cast_params.m_rayIncrement = 0.8f * hash_params.truncation_distance;
  ray_cast_params.m_thresSampleDist = 50.5f * ray_cast_params.m_rayIncrement;
  ray_cast_params.m_thresDist = 50.0f * ray_cast_params.m_rayIncrement;
  bool m_useGradients = true;

  RayCaster ray_caster(ray_cast_params);
  mapper.BindSensorDataToTexture(sensor.getSensorData());

  /// Process
  cv::Mat depth = cv::imread(depth_img_list[0], -1);
  cv::Mat color = cv::imread(color_img_list[0]);
  cv::cvtColor(color, color, CV_BGR2BGRA);
  // cv::imshow("test", color); cv::waitKey(-1);
  /// input cpu data OK


  sensor.process(depth, color);
  // checkCudaFloatMemory(sensor.getSensorData().depth_image_);
  // checkCudaFloat4Memory(sensor.getSensorData().color_image_);
  /// input gpu data OK

  LOG(INFO) << "Integrate";
  mapper.Integrate(T, sensor.getSensorData(), sensor.getSensorParams(), NULL);
  mapper.Integrate(T, sensor.getSensorData(), sensor.getSensorParams(), NULL);
  mapper.Integrate(T, sensor.getSensorData(), sensor.getSensorParams(), NULL);

  //mapper.debugHash();
  /// seems ok
  /// output blocks seems correct

  LOG(INFO) << "Render";
  ray_caster.Cast(mapper.getHashTable(), mapper.getHashParams(),
                    sensor.getSensorData(), T);
  /// runnable, still have bugs

  float4 *cuda_hsv;
  checkCudaErrors(cudaMalloc(&cuda_hsv, sizeof(float4) * 640 * 480));
  DepthToRGBCudaHost(cuda_hsv, ray_caster.ray_caster_data().depth_image_, 640, 480, 0.5f, 3.5f);
  checkCudaFloat4Memory(cuda_hsv);
  //checkCudaFloat4Memory(ray_caster.ray_caster_data().color_image_);
}