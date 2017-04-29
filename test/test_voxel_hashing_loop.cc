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

#define ICL
#ifdef ICL
const std::string kDefaultDatasetPath = "/home/wei/data/ICL/kt2/";

void DepthToRGBCudaHost(float4* d_output, float* d_input,
                unsigned int width, unsigned int height,
                float minDepth, float maxDepth);

void LoadImageList(std::string dataset_path, std::string dataset_txt,
                   std::vector<std::string> &image_name_list) {
  std::ifstream list_stream(dataset_path + "associations.txt");
  std::string file_name, rgb_file_name;

  std::string time_stamp;
  while (list_stream >> time_stamp >> file_name
                     >> time_stamp >> rgb_file_name) {
    image_name_list.push_back(dataset_path + "/" + file_name);
  }
}

void LoadTrajectory(std::string dataset_path, std::vector<float4x4> &wTc_list) {
  std::ifstream list_stream(dataset_path + "trajectory.txt");
  std::string ts_img, img_path, ts_gt;
  float tx, ty, tz, qx, qy, qz, qw;
  while (list_stream >> ts_img //>> img_path >> ts_gt
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
#else
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
  hash_params.entry_count = hash_params.bucket_count * hash_params.bucket_size;
  hash_params.linked_list_size = 7;

  hash_params.value_capacity = 1000000;
  hash_params.block_size = 8;
  hash_params.voxel_count = hash_params.value_capacity
                            * (hash_params.block_size * hash_params.block_size * hash_params.block_size);
  hash_params.voxel_size = 0.01;

  hash_params.sdf_upper_bound = 4.0;
  hash_params.truncation_distance_scale = 0.01;
  hash_params.truncation_distance = 0.02;
  hash_params.weight_sample = 10;
  hash_params.weight_upper_bound = 255;
  SetConstantHashParams(hash_params);
  Map voxel_map(hash_params);

  //mapper.debugHash();
  /// Only to alloc cuda memory, suppose its ok

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
  SetConstantSensorParams(sensor_params);
  Sensor sensor(sensor_params);
  sensor.BindSensorDataToTexture();

  float4x4 T;
  float4x4 K;
  K.m11 = sensor_params.fx;
  K.m13 = sensor_params.cx;
  K.m22 = sensor_params.fy;
  K.m23 = sensor_params.cy;

  /// Ray Caster
  RayCasterParams ray_cast_params;
  ray_cast_params.intrinsics = K;
  ray_cast_params.intrinsics_inverse = K.getInverse();
  ray_cast_params.width = 640;
  ray_cast_params.height = 480;
  ray_cast_params.min_raycast_depth = 0.5f;
  ray_cast_params.max_raycast_depth = 5.5f;
  ray_cast_params.raycast_step = 0.8f * hash_params.truncation_distance;
  ray_cast_params.sample_sdf_threshold = 50.5f * ray_cast_params.raycast_step;
  ray_cast_params.sdf_threshold = 50.0f * ray_cast_params.raycast_step;
  SetConstantRayCasterParams(ray_cast_params);
  RayCaster ray_caster(ray_cast_params);

  Mapper mapper;

  /// Process
  float4 *cuda_hsv;
  checkCudaErrors(cudaMalloc(&cuda_hsv, sizeof(float4) * 640 * 480));

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
    T = wTc[0].getInverse() * wTc[i];
    sensor.set_transform(T);
    //T = T.getInverse();

    mapper.Integrate(&voxel_map, &sensor, NULL);

    ray_caster.Cast(&voxel_map, T.getInverse());
    //DepthToRGBCudaHost(cuda_hsv, ray_caster.ray_caster_data().depth_image_, 640, 480, 0.5f, 3.5f);
    //checkCudaFloat4Memory(cuda_hsv);
    checkCudaFloat4Memory(ray_caster.ray_caster_data().normal_image_);

    cv::waitKey(1);
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  LOG(INFO) << "Total time: " << seconds.count();
  LOG(INFO) << "Fps: " << frames / seconds.count();

  checkCudaErrors(cudaFree(cuda_hsv));
  //cv::waitKey(-1);
  voxel_map.hash_table_.Debug();
  /// seems ok
  /// output blocks seems correct

  LOG(INFO) << "Render";

  //checkCudaFloat4Memory(ray_caster.ray_caster_data().color_image_);
}