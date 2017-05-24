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
#include "control.h"

#define ICL
#if defined(ICL)
const std::string kDefaultDatasetPath = "/home/wei/data/ICL/lv1/";
#elif defined(TUM)
const std::string kDefaultDatasetPath =
        "/home/wei/data/TUM/rgbd_dataset_freiburg3_long_office_household/";
#elif defined(SUN3D)
const std::string kDefaultDatasetPath =
        "/home/wei/data/SUN3D/copyroom/";
#elif defined(SUN3D_ORI)
const std::string kDefaultDatasetPath =
        "/home/wei/data/SUN3D-Princeton/hotel_umd/maryland_hote`l3/";
#elif defined(TDVCR)
const std::string kDefaultDatasetPath =
        "/home/wei/data/3DVCR/hall2/";
#endif

/// Refer to constant.cu
extern void SetConstantSDFParams(const SDFParams& params);

int main(int argc, char** argv) {
  /// Load images
  std::vector<std::string> depth_img_list;
  std::vector<std::string> color_img_list;
  std::vector<float4x4>    wTc;

  ConfigReader config;
#if defined(ICL)
  LoadICL(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/icl.yml");
#elif defined(TUM)
  LoadTUM(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/tum3.yml");
#elif defined(SUN3D)
  LoadSUN3D(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/sun3d.yml");
#elif defined(SUN3D_ORI)
  LoadSUN3DOriginal(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/sun3d_ori.yml");
#elif defined(TDVCR)
  Load3DVCR(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/3dvcr.yml");
#endif

  std::vector<std::string> uniform_names;
  MeshRenderer mesh_renderer("Mesh",
                             config.ray_caster_params.width,
                             config.ray_caster_params.height);
  mesh_renderer.free_walk() = false;

  /// Support only one GL instance yet
  uniform_names.clear();
  uniform_names.push_back("mvp");
  mesh_renderer.CompileShader("../shader/mesh_vertex.glsl",
                              "../shader/mesh_fragment.glsl",
                              uniform_names);
  SetConstantSDFParams(config.sdf_params);

  Map voxel_map(config.hash_params);
  LOG(INFO) << "Map allocated";

  Sensor sensor(config.sensor_params);
  sensor.BindGPUTexture();

  RayCaster ray_caster(config.ray_caster_params);

//  cv::VideoWriter writer("icl-vh.avi", CV_FOURCC('X','V','I','D'),
//                         30, cv::Size(640, 480));

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int frames = depth_img_list.size() - 1;

  for (int i = 0; i < frames; ++i) {
    LOG(INFO) << i;
    cv::Mat depth = cv::imread(depth_img_list[i], -1);
    cv::Mat color = cv::imread(color_img_list[i]);

    cv::cvtColor(color, color, CV_BGR2BGRA);

    sensor.Process(depth, color);
    float4x4 c0Tc = wTc[0].getInverse() * wTc[i];
    sensor.set_transform(c0Tc);

    voxel_map.Integrate(sensor, NULL);
    voxel_map.MarchingCubes();

//    ray_caster.Cast(voxel_map, c0Tc.getInverse());
//    cv::imshow("display", ray_caster.normal_image());
//    cv::waitKey(1);

    //voxel_map.CollectAllBlocks();
    voxel_map.CompressMesh();
    mesh_renderer.Render(voxel_map.compact_mesh().vertices(),
                         (size_t)voxel_map.compact_mesh().vertex_count(),
                         voxel_map.compact_mesh().normals(),
                         (size_t)voxel_map.compact_mesh().vertex_count(),
                         voxel_map.compact_mesh().triangles(),
                         (size_t)voxel_map.compact_mesh().triangle_count(),
                         c0Tc.getInverse());
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  LOG(INFO) << "Total time: " << seconds.count();
  LOG(INFO) << "Fps: " << frames / seconds.count();

  voxel_map.SaveMesh("kkk.obj");
  return 0;
}