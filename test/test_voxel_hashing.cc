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

#include "renderer.h"

#include "config_reader.h"

#define TDVCR
#if defined(ICL)
const std::string kDefaultDatasetPath = "/home/wei/data/ICL/lv1/";
#elif defined(TUM)
const std::string kDefaultDatasetPath =
        "/home/wei/data/TUM/rgbd_dataset_freiburg3_long_office_household/";
#elif defined(SUN3D)
const std::string kDefaultDatasetPath =
        "/home/wei/data/SUN3D/lounge/";
#elif defined(SUN3D_ORI)
const std::string kDefaultDatasetPath =
        "/home/wei/data/SUN3D-Princeton/hotel_umd/maryland_hotel3/";
#elif defined(TDVCR)
const std::string kDefaultDatasetPath =
        "/home/wei/data/3DVCR/lab1/";
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
  static float cpu_memory[640 * 360 * 4];
  cv::Mat matf = cv::Mat(360, 640, CV_32FC4, cpu_memory);

  checkCudaErrors(cudaMemcpy(cpu_memory, cuda_memory,
                             sizeof(float) * 4 * 640 * 360,
                             cudaMemcpyDeviceToHost));

#define WRITE
#ifdef WRITE
  cv::Mat matb = cv::Mat(360, 640, CV_8UC3);
  for (int i = 0; i < 360; ++i) {
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

/// Refer to constant.cu
extern void SetConstantSDFParams(const SDFParams& params);

int main(int argc, char** argv) {
  /// Load images
  std::vector<std::string> depth_img_list;
  std::vector<std::string> color_img_list;
  std::vector<float4x4>    wTc;

#if defined(ICL)
  LoadICL(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
#elif defined(TUM)
  LoadTUM(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
#elif defined(SUN3D)
  LoadSUN3D(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
#elif defined(SUN3D_ORI)
  LoadSUN3DOriginal(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
#elif defined(TDVCR)
  Load3DVCR(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
#endif

  SDFParams sdf_params;
  LoadSDFParams("../config/sdf.yml", sdf_params);
  SetConstantSDFParams(sdf_params);

  HashParams hash_params;
  LoadHashParams("../config/hash.yml", hash_params);

  SensorParams sensor_params;

#if defined(ICL)
  LoadSensorParams("../config/sensor_icl.yml", sensor_params);
#elif defined(TUM)
  LoadSensorParams("../config/sensor_tum3.yml", sensor_params);
#elif defined(SUN3D)
  LoadSensorParams("../config/sensor_sun3d.yml", sensor_params);
#elif defined(SUN3D_ORI)
  LoadSensorParams("../config/sensor_sun3d_ori.yml", sensor_params);
#elif defined(TDVCR)
  LoadSensorParams("../config/sensor_3dvcr.yml", sensor_params);
#endif

  RayCasterParams ray_cast_params;
  LoadRayCasterParams("../config/ray_caster.yml", ray_cast_params);
  ray_cast_params.fx = sensor_params.fx;
  ray_cast_params.fy = sensor_params.fy;
  ray_cast_params.cx = sensor_params.cx;
  ray_cast_params.cy = sensor_params.cy;

  Map voxel_map(hash_params);
  LOG(INFO) << "map allocated";

  Sensor sensor(sensor_params);
  sensor.BindGPUTexture();

  RayCaster ray_caster(ray_cast_params);

  //cv::VideoWriter writer("icl-vh.avi", CV_FOURCC('X','V','I','D'),
  //                       30, cv::Size(640, 480));

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int frames = depth_img_list.size() - 1;

  for (int i = 0; i < frames; ++i) {
    LOG(INFO) << i;
    cv::Mat depth = cv::imread(depth_img_list[i], -1);
    cv::Mat color = cv::imread(color_img_list[i]);

    cv::cvtColor(color, color, CV_BGR2BGRA);

    sensor.Process(depth, color);
    float4x4 T = wTc[0].getInverse() * wTc[i];
    sensor.set_transform(T);

    voxel_map.Integrate(sensor, NULL);
    voxel_map.MarchingCubes();
    voxel_map.CompressMesh();

    if (i > 0 && i % 500 == 0) {
      std::stringstream ss;
      ss.str("");
      ss << "model-" << i << ".obj";
      voxel_map.SaveMesh(ss.str());
    }

    ray_caster.Cast(voxel_map, T.getInverse());
    cv::Mat display = GPUFloat4ToMat(ray_caster.gpu_data().normal_image);
    cv::imshow("display", display);
    cv::waitKey(1);
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  LOG(INFO) << "Total time: " << seconds.count();
  LOG(INFO) << "Fps: " << frames / seconds.count();

  voxel_map.SaveMesh("test20.obj");
}