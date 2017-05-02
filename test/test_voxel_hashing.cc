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
#include <mesh.h>

#include "fuser.h"
#include "renderer.h"

#include "config_reader.h"

#define ICL
#ifdef ICL
const std::string kDefaultDatasetPath = "/home/wei/data/ICL/kt2/";
#else
const std::string kDefaultDatasetPath = "/home/wei/data/TUM/rgbd_dataset_freiburg2_xyz/";
#endif

/// Only test over 480x640 images
cv::Mat GPUFloatToMat(float* cuda_memory) {
  static float cpu_memory[640 * 480];
  cv::Mat matf = cv::Mat(480, 640, CV_32F, cpu_memory);
  checkCudaErrors(cudaMemcpy(cpu_memory, cuda_memory,
                             sizeof(float) * 640 * 480,
                             cudaMemcpyDeviceToHost));
  return matf;
}
cv::Mat GPUFloat4ToMat(float4 *cuda_memory) {
  static float cpu_memory[640 * 480 * 4];
  cv::Mat matf = cv::Mat(480, 640, CV_32FC4, cpu_memory);

  checkCudaErrors(cudaMemcpy(cpu_memory, cuda_memory,
                             sizeof(float) * 4 * 640 * 480,
                             cudaMemcpyDeviceToHost));

#define WRITE
#ifdef WRITE
  cv::Mat matb = cv::Mat(480, 640, CV_8UC3);
  for (int i = 0; i < 480; ++i) {
    for (int j = 0; j < 640; ++j) {
      cv::Vec4f cf = matf.at<cv::Vec4f>(i, j);
      if (std::isinf(cf[0])) {
        matb.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
      } else {
        matb.at<cv::Vec3b>(i, j) = cv::Vec3b(255 * fabs(cf[0]),
                                             255 * fabs(cf[1]),
                                             255 * fabs(cf[2]));
      }
    }
  }
  return matb;
#else
  return matf;
#endif
}

extern void SetConstantSDFParams(const SDFParams& params);

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
  LoadSDFParams("../config/sdf.yml", sdf_params);
  SetConstantSDFParams(sdf_params);

  HashParams hash_params;
  LoadHashParams("../config/hash.yml", hash_params);

  SensorParams sensor_params;
#ifdef ICL
  LoadSensorParams("../config/sensor_icl.yml", sensor_params);
#else
  LoadSensorParams("../config/sensor_tum.yml", sensor_params);
#endif

  RayCasterParams ray_cast_params;
  LoadRayCasterParams("../config/ray_caster.yml", ray_cast_params);
  ray_cast_params.fx = sensor_params.fx;
  ray_cast_params.fy = sensor_params.fy;
  ray_cast_params.cx = sensor_params.cx;
  ray_cast_params.cy = sensor_params.cy;

  Map voxel_map(hash_params);
  LOG(INFO) << "map allocated";

  Mesh mesh(hash_params);
  //LOG(INFO) << "mesh allocated";

  Sensor sensor(sensor_params);
  sensor.BindSensorDataToTexture();

  Fuser fuser;

  RayCaster ray_caster(ray_cast_params);

  //cv::VideoWriter writer("icl-vh.avi", CV_FOURCC('X','V','I','D'),
  //                       30, cv::Size(640, 480));
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int frames = 2;
  for (int i = 0; i < frames; ++i) {
    LOG(INFO) << i;
    cv::Mat depth = cv::imread(depth_img_list[i], -1);
    cv::Mat color = cv::imread(color_img_list[i]);
    cv::cvtColor(color, color, CV_BGR2BGRA);

    sensor.Process(depth, color);
    float4x4 T = wTc[0].getInverse() * wTc[i];
    sensor.set_transform(T);

    fuser.Integrate(&voxel_map, &mesh, &sensor, NULL);
    mesh.MarchingCubes(&voxel_map);

    //ray_caster.Cast(&voxel_map, T.getInverse());
    //cv::Mat display = GPUFloat4ToMat(ray_caster.ray_caster_data().normal_image);
    //cv::imshow("display", display);
    //cv::waitKey(1);
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  LOG(INFO) << "Total time: " << seconds.count();
  LOG(INFO) << "Fps: " << frames / seconds.count();

  //mesh.MarchingCubes(&voxel_map);
  mesh.SaveMesh("test.obj");




  voxel_map.Debug();
}