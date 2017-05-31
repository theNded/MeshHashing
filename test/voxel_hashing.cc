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

#include "dataset_manager.h"

const Dataset datasets[] = {
        {ICL,            "/home/wei/data/ICL/lv1/"},
        {TUM1,           "/home/wei/data/TUM/rgbd_dataset_freiburg1_xyz/"},
        {TUM2,           "/home/wei/data/TUM/rgbd_dataset_freiburg2_xyz/"},
        {TUM3,           "/home/wei/data/TUM/rgbd_dataset_freiburg3_long_office_household/"},
        {SUN3D,          "/home/wei/data/SUN3D/copyroom/"},
        {SUN3D_ORIGINAL, "/home/wei/data/SUN3D-Princeton/hotel_umd/maryland_hotel3/"},
        {PKU,            "/home/wei/data/3DVCR/hall2/"}
};

/// Refer to constant.cu
extern void SetConstantSDFParams(const SDFParams& params);

int main(int argc, char** argv) {
  ConfigManager config;
  DataManager   rgbd_data;

  DatasetType dataset_type = TUM2;

  /// Probably the path will change
  config.LoadConfig(dataset_type);
  rgbd_data.LoadDataset(datasets[dataset_type]);

  std::vector<std::string> uniform_names;
  MeshRenderer mesh_renderer("Mesh",
                             config.sensor_params.width,
                             config.sensor_params.height,
                             config.mesh_params.max_vertex_count,
                             config.mesh_params.max_triangle_count);
  mesh_renderer.free_walk() = true;
  mesh_renderer.line_only() = true;
  mesh_renderer.new_mesh_only() = false;

  /// Support only one GL instance yet
  uniform_names.clear();
  uniform_names.push_back("mvp");
  mesh_renderer.CompileShader("../shader/mesh_vertex.glsl",
                              "../shader/mesh_fragment.glsl",
                              uniform_names);
  SetConstantSDFParams(config.sdf_params);

  Map voxel_map(config.hash_params, config.mesh_params);
  LOG(INFO) << "Map allocated";

  Sensor sensor(config.sensor_params);

  RayCaster ray_caster(config.ray_caster_params);

//  cv::VideoWriter writer("icl-vh.avi", CV_FOURCC('X','V','I','D'),
//                         30, cv::Size(640, 480));

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int frames = rgbd_data.depth_image_list.size() - 1;

  //cv::Mat capture = cv::Mat(480, 640, CV_8UC3);

  cv::Mat color, depth;
  float4x4 cTw;
  for (int i = 0; i < frames; ++i) {
    LOG(INFO) << i;
    rgbd_data.ProvideData(depth, color, cTw);
    sensor.Process(depth, color);
    sensor.set_transform(cTw);

    voxel_map.Integrate(sensor, NULL);
    voxel_map.MarchingCubes();

    ray_caster.Cast(voxel_map, cTw.getInverse());
    cv::imshow("display", ray_caster.normal_image());
    cv::waitKey(1);

    if (! mesh_renderer.new_mesh_only()) {
      voxel_map.CollectAllBlocks();
    }
    voxel_map.CompressMesh();
    mesh_renderer.Render(voxel_map.compact_mesh().vertices(),
                         (size_t)voxel_map.compact_mesh().vertex_count(),
                         voxel_map.compact_mesh().normals(),
                         (size_t)voxel_map.compact_mesh().vertex_count(),
                         voxel_map.compact_mesh().triangles(),
                         (size_t)voxel_map.compact_mesh().triangle_count(),
                         cTw.getInverse());

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = end - start;
    LOG(INFO) << "Fps: " << (i + 1) / seconds.count();
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  LOG(INFO) << "Total time: " << seconds.count();
  LOG(INFO) << "Fps: " << frames / seconds.count();

  voxel_map.SaveMesh("kkk.obj");

  return 0;
}