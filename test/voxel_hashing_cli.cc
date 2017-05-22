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

/// Refer to constant.cu
extern void SetConstantSDFParams(const SDFParams& params);

const std::string kDatasetTypes[] = {
        "icl", "tum", "sun3d", "sun3d_origin"
};

enum DatasetType {
  kNULL        = -1,
  kICL         = 0,
  kTUM         = 1,
  kSUN3D       = 2,
  kSUN3DOrigin = 3,
  kTypes       = 4
};

int main(int argc, char** argv) {
  CHECK(argc == 4) << "Usage: ./voxel_hashing "
          "dataset_type=[icl, tum, sun3d, sun3d_origin] "
          "dataset_path=/path/to/dataset "
          "obj_path=/path/to/obj";

  std::string dataset_type = std::string(argv[1]);
  std::string dataset_path = std::string(argv[2]);
  std::string obj_path     = std::string(argv[3]);
  int type = kNULL;
  for (int i = 0; i < kTypes; ++i) {
    if (kDatasetTypes[i] == dataset_type) {
      type = i;
    }
  }

  /// Load images
  std::vector<std::string> depth_img_list;
  std::vector<std::string> color_img_list;
  std::vector<float4x4>    wTc;

  /// Load files and sensor params
  SensorParams sensor_params;
  switch (type) {
    case kICL:
      LoadICL(dataset_path, depth_img_list, color_img_list, wTc);
      LoadSensorParams("../config/sensor_icl.yml", sensor_params);
      break;

    case kTUM:
      LoadTUM(dataset_path, depth_img_list, color_img_list, wTc);
      LoadSensorParams("../config/sensor_tum3.yml", sensor_params);
      break;

    case kSUN3D:
      LoadSUN3D(dataset_path, depth_img_list, color_img_list, wTc);
      LoadSensorParams("../config/sensor_sun3d.yml", sensor_params);
      break;

    case kSUN3DOrigin:
      LoadSUN3DOriginal(dataset_path, depth_img_list, color_img_list, wTc);
      LoadSensorParams("../config/sensor_sun3d_ori.yml", sensor_params);
      break;

    default:
      LOG(ERROR) << "Unknwon data type, exit";
      return -1;
  }

  CHECK(depth_img_list.size() > 0 && color_img_list.size() > 0
        && wTc.size() > 0) << "invalid data";

  /// Load other configs
  SDFParams sdf_params;
  LoadSDFParams("../config/sdf.yml", sdf_params);
  SetConstantSDFParams(sdf_params);

  HashParams hash_params;
  LoadHashParams("../config/hash.yml", hash_params);

  RayCasterParams ray_cast_params;
  LoadRayCasterParams("../config/ray_caster.yml", ray_cast_params);
  ray_cast_params.fx = sensor_params.fx;
  ray_cast_params.fy = sensor_params.fy;
  ray_cast_params.cx = sensor_params.cx;
  ray_cast_params.cy = sensor_params.cy;

  /// Core
  Map voxel_map(hash_params);
  LOG(INFO) << "Map successfully allocated";

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
      ss << obj_path << "_" << i << ".obj";
      voxel_map.SaveMesh(ss.str());
    }

    ray_caster.Cast(voxel_map, T.getInverse());
    cv::Mat n = GPUFloat4ToMat(ray_caster.gpu_data().normal_image);
    cv::imshow("normal", n);
    cv::waitKey(1);
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  LOG(INFO) << "Total time: " << seconds.count();
  LOG(INFO) << "Fps: " << frames / seconds.count();

  voxel_map.SaveMesh(obj_path + ".obj");
}