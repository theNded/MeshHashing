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

#include "util/timer.h"
#include <queue>
#include <engine/visualizing_engine.h>
#include <io/mesh_writer.h>
#include <meshing/marching_cubes.h>
#include <visualization/compress_mesh.h>

#include "sensor/rgbd_data_provider.h"
#include "sensor/rgbd_sensor.h"
#include "visualization/ray_caster.h"

#include "io/config_manager.h"
#include "core/collect_block_array.h"
#include "glwrapper.h"

#define DEBUG_

Light light = {
  {
      glm::vec3(0, -2, 0),
      glm::vec3(4, -2, 0)
  },
  glm::vec3(1, 1, 1),
  3.0f
};

int main(int argc, char** argv) {
  /// Use this to substitute tedious argv parsing
  RuntimeParams args;
  LoadRuntimeParams("../config/args.yml", args);

  ConfigManager config;
  RGBDDataProvider rgbd_local_sequence;

  DatasetType dataset_type = DatasetType(args.dataset_type);
  config.LoadConfig(dataset_type);
  rgbd_local_sequence.LoadDataset(dataset_type);
  Sensor    sensor(config.sensor_params);

  MainEngine main_engine(config.hash_params,
                         config.mesh_params,
                         config.sdf_params);
  main_engine.ConfigVisualizingEngineMesh(light,
                                          args.enable_navigation,
                                          args.enable_global_mesh,
                                          args.enable_bounding_box,
                                          args.enable_trajectory,
                                          args.enable_polygon_mode);
  if (args.enable_ray_casting) {
    main_engine.ConfigVisualizingEngineRaycaster(config.ray_caster_params);
  }
  if (args.enable_video_recording) {
    main_engine.ConfigLoggingEngine(".",
                                    args.enable_video_recording,
                                    args.enable_ply_saving);
  }
  main_engine.use_fine_gradient() = args.fine_gradient;

  cv::Mat color, depth;
  float4x4 wTc, cTw;
  int frame_count = 0;
  while (rgbd_local_sequence.ProvideData(depth, color, wTc)) {
    frame_count ++;
    if (args.run_frames > 0 &&  frame_count > args.run_frames)
      break;

    // Preprocess data
    sensor.Process(depth, color);
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();

    main_engine.Mapping(sensor);
    main_engine.Meshing();
    main_engine.Visualize(cTw);

    main_engine.Log();
    main_engine.Recycle();
  }

  main_engine.FinalLog();

  return 0;
}