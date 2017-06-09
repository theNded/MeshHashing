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
#include <debugger.h>

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

  MeshType mesh_type = args.render_type == 0 ? kNormal : kColor;

  Renderer renderer("Mesh",
                    config.sensor_params.width,
                    config.sensor_params.height);
  MeshObject mesh(config.mesh_params.max_vertex_count,
                  config.mesh_params.max_triangle_count,
                  mesh_type);
  mesh.line_only() = args.line_only;
  renderer.AddObject(&mesh);

  LineObject* bbox;
  if (args.bounding_box) {
    bbox = new LineObject(config.hash_params.value_capacity * 24);
    renderer.AddObject(bbox);
  }

  renderer.free_walk() = args.free_walk;

  SetConstantSDFParams(config.sdf_params);
  Map       map(config.hash_params, config.mesh_params);
  Sensor    sensor(config.sensor_params);
  RayCaster ray_caster(config.ray_caster_params);

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

  std::chrono::time_point<std::chrono::system_clock> start, end;

#ifdef DEBUG
  Debugger debugger(config.hash_params.entry_count,
                    config.hash_params.value_capacity);
#endif
  while (rgbd_data.ProvideData(depth, color, wTc)) {
    start = std::chrono::system_clock::now();

    frame_count ++;
    if (args.run_frames > 0
        &&  frame_count > args.run_frames)
      break;

    sensor.Process(depth, color);
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();

    map.Integrate(sensor);
    map.MarchingCubes();

#ifdef DEBUG
    debugger.CoreDump(map.hash_table().gpu_data());
    debugger.CoreDump(map.blocks().gpu_data());
    debugger.DebugHashToBlock();
#endif

    if (args.ray_casting) {
      ray_caster.Cast(map, cTw);
      cv::imshow("RayCasting", ray_caster.surface_image());
      cv::waitKey(1);
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = end - start;
    LOG(INFO) << "Total time: " << seconds.count();
    LOG(INFO) << "Fps: " << 1.0f / seconds.count();

    if (! args.new_mesh_only) {
      map.CollectAllBlocks();
    }
    map.GetBoundingBoxes();
    map.CompressMesh();

    if (args.bounding_box) {
      bbox->SetData(map.bbox().vertices(),
                    map.bbox().vertex_count());
    }
    mesh.SetData(map.compact_mesh().vertices(),
                 (size_t)map.compact_mesh().vertex_count(),
                 args.render_type == 0 ?
                 map.compact_mesh().normals() : map.compact_mesh().colors(),
                 (size_t)map.compact_mesh().vertex_count(),
                 map.compact_mesh().triangles(),
                 (size_t)map.compact_mesh().triangle_count());
    renderer.Render(cTw);

    if (args.record_video) {
      renderer.ScreenCapture(screen.data, screen.cols, screen.rows);
      cv::flip(screen, screen, 0);
      writer << screen;
    }
  }

#ifdef DEBUG
  debugger.PrintDebugInfo();
#endif
  if (args.save_mesh) {
    map.SaveMesh(args.filename_prefix + ".obj");
  }

  return 0;
}