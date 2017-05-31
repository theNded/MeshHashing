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
#include "datasets.h"


/// Refer to constant.cu
extern void SetConstantSDFParams(const SDFParams& params);

int main(int argc, char** argv) {
  /// Use this to substitute tedious argv parsing
  RuntimeParams args;
  LoadRuntimeParams("../config/args.yml", args);

  ConfigManager config;
  DataManager   rgbd_data;

  DatasetType dataset_type = DatasetType(args.dataset_type);
  config.LoadConfig(dataset_type);
  rgbd_data.LoadDataset(datasets[dataset_type]);

  MapMeshRenderer mesh_renderer("Mesh",
                                config.sensor_params.width,
                                config.sensor_params.height,
                                config.mesh_params.max_vertex_count,
                                config.mesh_params.max_triangle_count);

  SetConstantSDFParams(config.sdf_params);
  Map       map(config.hash_params, config.mesh_params);
  Sensor    sensor(config.sensor_params);
  RayCaster ray_caster(config.ray_caster_params);

  mesh_renderer.free_walk() = args.free_walk;
  mesh_renderer.line_only() = args.line_only;
  map.use_fine_gradient()   = args.fine_gradient;

  cv::VideoWriter writer;
  cv::Mat screen;
  if (args.record_video) {
    writer = cv::VideoWriter(args.filename_prefix + ".avi",
                             CV_FOURCC('X','V','I','D'),
                             30, cv::Size(config.sensor_params.width,
                                          config.sensor_params.height));
    screen = cv::Mat(config.sensor_params.height,
                     config.sensor_params.width,
                     CV_8UC3);
  }

  cv::Mat color, depth;
  float4x4 wTc, cTw;
  int frame_count = 0;
  while (rgbd_data.ProvideData(depth, color, wTc)) {
    if (args.run_frames > 0
        && frame_count ++ > args.run_frames)
      break;
    sensor.Process(depth, color);
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();

    map.Integrate(sensor);
    map.MarchingCubes();

    if (args.ray_casting) {
      ray_caster.Cast(map, cTw);
      cv::imshow("RayCasting", ray_caster.normal_image());
      cv::waitKey(1);
    }

    if (! args.new_mesh_only) {
      map.CollectAllBlocks();
    }
    map.CompressMesh();
    mesh_renderer.Render(map.compact_mesh().vertices(),
                         (size_t)map.compact_mesh().vertex_count(),
                         map.compact_mesh().normals(),
                         (size_t)map.compact_mesh().vertex_count(),
                         map.compact_mesh().triangles(),
                         (size_t)map.compact_mesh().triangle_count(),
                         cTw);

    if (args.record_video) {
      mesh_renderer.ScreenCapture(screen.data, screen.cols, screen.rows);
      cv::flip(screen, screen, 0);
      writer << screen;
    }
  }

  if (args.save_mesh) {
    map.SaveMesh(args.filename_prefix + ".obj");
  }
  return 0;
}