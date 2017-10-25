/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>

#include <System.h>
#include <glog/logging.h>
#include <sensor/rgbd_data_provider.h>

#include "core/params.h"
#include "io/config_manager.h"
#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"
#include "visualization/ray_caster.h"
#include "glwrapper.h"

const std::string orb_configs[] = {
    "../config/ORB/ICL.yaml",
    "../config/ORB/TUM1.yaml",
    "../config/ORB/TUM2.yaml",
    "../config/ORB/TUM3.yaml",
    "../config/ORB/SUN3D.yaml",
    "../config/ORB/SUN3D_ORIGINAL.yaml",
    "../config/ORB/PKU.yaml"
};

std::string path_to_vocabulary = "../src/extern/orb_slam2/Vocabulary/ORBvoc.bin";

Light light = {
    {
        glm::vec3(0, -2, 0),
        glm::vec3(4, -2, 0)
    },
    glm::vec3(1, 1, 1),
    3.0f
};

float4x4 MatTofloat4x4(cv::Mat m) {
  float4x4 T;
  T.setIdentity();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      T.entries2[i][j] = (float)m.at<float>(i, j);
  return T;
}

int main(int argc, char **argv) {
  /// Use this to substitute tedious argv parsing
  RuntimeParams args;
  LoadRuntimeParams("../config/args.yml", args);

  ConfigManager config;
  RGBDDataProvider   rgbd_local_sequence;

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
  main_engine.enable_sdf_gradient() = args.enable_sdf_gradient;

  ORB_SLAM2::System orb_slam_engine(path_to_vocabulary,
                         orb_configs[dataset_type],
                         ORB_SLAM2::System::RGBD,
                         true);

  double tframe;
  cv::Mat color, depth;
  float4x4 wTc, cTw;
  int frame_count = 0;
  while (rgbd_local_sequence.ProvideData(depth, color, wTc)) {
    frame_count ++;
    if (args.run_frames > 0 && frame_count > args.run_frames)
      break;

    cv::Mat color_slam = color.clone();
    cv::Mat depth_slam = depth.clone();
    cv::Mat cTw_orb = orb_slam_engine.TrackRGBD(color_slam, depth_slam, tframe);
    if (cTw_orb.empty()) continue;
    cTw = MatTofloat4x4(cTw_orb);
    wTc = cTw.getInverse();

    sensor.Process(depth, color); // abandon wTc
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();

    main_engine.Mapping(sensor);
    main_engine.Meshing();
    main_engine.Visualize(cTw);

    main_engine.Log();
    main_engine.Recycle();
  }

  main_engine.FinalLog();
  orb_slam_engine.Shutdown();
  return 0;
}