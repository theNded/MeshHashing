//
// Created by wei on 17-3-26.
//
#include <string>
#include <vector>
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
#include <timer.h>
#include "../tool/cpp/debugger.h"

#include "renderer.h"

#include "dataset_manager.h"

#define DEBUG

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
  rgbd_data.LoadDataset(dataset_type);

  MeshType mesh_type = args.render_type == 0 ? kNormal : kColor;

  Renderer renderer("Mesh",
                    config.sensor_params.width,
                    config.sensor_params.height);
  MeshObject mesh(config.mesh_params.max_vertex_count,
                  config.mesh_params.max_triangle_count,
                  mesh_type);
  mesh.ploygon_mode() = args.ploygon_mode;
  renderer.AddObject(&mesh);

  LineObject* bbox;
  if (args.bounding_box) {
    bbox = new LineObject(config.hash_params.value_capacity * 24);
    renderer.AddObject(bbox);
  }

  renderer.free_walk() = args.free_walk;

  SetConstantSDFParams(config.sdf_params);
  Map       map(config.hash_params, config.mesh_params, "../result/statistics/" + args.time_profile + ".txt");
  Sensor    sensor(config.sensor_params);
  RayCaster ray_caster(config.ray_caster_params);

  map.use_fine_gradient()   = args.fine_gradient;

  cv::VideoWriter writer;
  cv::Mat screen;
  if (args.record_video) {
    writer = cv::VideoWriter("../result/videos/" + args.filename_prefix + ".avi",
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

  std::chrono::time_point<std::chrono::system_clock> start, end;

#ifdef DEBUG
  Debugger debugger(config.hash_params.entry_count,
                    config.hash_params.value_capacity,
                    config.mesh_params.max_vertex_count,
                    config.mesh_params.max_triangle_count,
                    config.sdf_params.voxel_size);
#endif
  while (rgbd_data.ProvideData(depth, color, wTc)) {
    Timer timer;
    timer.Tick();

    frame_count ++;
    if (args.run_frames > 0 &&  frame_count > args.run_frames)
      break;

    sensor.Process(depth, color);
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();

    map.Integrate(sensor);
    map.MarchingCubes();

    if (args.ray_casting) {
      ray_caster.Cast(map, cTw);
      cv::imshow("RayCasting", ray_caster.surface_image());
      cv::waitKey(1);
    }
    double seconds = timer.Tock();
    LOG(INFO) << "Total time: " << seconds;
    LOG(INFO) << "Fps: " << 1.0f / seconds;

    if (! args.mesh_range) {
      map.CollectAllBlocks();
    }
    map.GetBoundingBoxes();
    map.CompressMesh();

    if (args.bounding_box) {
      bbox->SetData(map.bbox().vertices(),
                    map.bbox().vertex_count());
    }

    if (args.render_type == 0) {
      mesh.SetData(map.compact_mesh().vertices(), map.compact_mesh().vertex_count(),
                   map.compact_mesh().normals(), map.compact_mesh().vertex_count(),
                   NULL, 0,
                   map.compact_mesh().triangles(), map.compact_mesh().triangle_count());
    } else {
      mesh.SetData(map.compact_mesh().vertices(), map.compact_mesh().vertex_count(),
                   NULL, 0,
                   map.compact_mesh().colors(), map.compact_mesh().vertex_count(),
                   map.compact_mesh().triangles(), map.compact_mesh().triangle_count());
    }
    renderer.Render(cTw);

    if (args.record_video) {
      renderer.ScreenCapture(screen.data, screen.cols, screen.rows);
      cv::flip(screen, screen, 0);
      writer << screen;
    }

  }

#ifdef DEBUG
  debugger.CoreDump(map.compact_hash_table().gpu_data());
  debugger.CoreDump(map.blocks().gpu_data());
  debugger.CoreDump(map.mesh().gpu_data());
  debugger.DebugAll();
#endif
  if (args.save_mesh) {
    map.SaveMesh("../result/models/" + args.filename_prefix + ".obj");
  }

  return 0;
}