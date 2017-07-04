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

#include "params.h"
#include "dataset_manager.h"
#include "map.h"
#include "sensor.h"
#include "ray_caster.h"
#include "renderer.h"

static const std::string orb_configs[] = {
    "../config/ORB/ICL.yaml",
    "../config/ORB/TUM1.yaml",
    "../config/ORB/TUM2.yaml",
    "../config/ORB/TUM3.yaml",
    "../config/ORB/SUN3D.yaml",
    "../config/ORB/SUN3D_ORIGINAL.yaml",
    "../config/ORB/PKU.yaml"
};

std::string path_to_vocabulary = "../../../opensource/orb_slam2/Vocabulary/ORBvoc.bin";

extern void SetConstantSDFParams(const SDFParams& params);

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
  DataManager   rgbd_data;

  DatasetType dataset_type = DatasetType(args.dataset_type);
  config.LoadConfig(dataset_type);
  rgbd_data.LoadDataset(dataset_type);

  Renderer mesh_renderer("Mesh",
                         config.sensor_params.width,
                         config.sensor_params.height);

  MeshObject mesh(config.mesh_params.max_vertex_count,
                  config.mesh_params.max_triangle_count);
  mesh_renderer.free_walk()     = args.free_walk;
  mesh.ploygon_mode()     = args.ploygon_mode;
  mesh_renderer.AddObject(&mesh);

  SetConstantSDFParams(config.sdf_params);
  Map       map(config.hash_params, config.mesh_params);
  Sensor    sensor(config.sensor_params);
  RayCaster ray_caster(config.ray_caster_params);

  map.use_fine_gradient()       = args.fine_gradient;

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

  ORB_SLAM2::System SLAM(path_to_vocabulary,
                         orb_configs[dataset_type],
                         ORB_SLAM2::System::RGBD,
                         true);

  cv::Mat color, depth;
  float4x4 wTc, cTw;
  double tframe;

  int frame_count = 0;
  while (rgbd_data.ProvideData(depth, color, wTc)) {
    if (args.run_frames > 0
        && frame_count ++ > args.run_frames)
      break;

    sensor.Process(depth, color); // abandon wTc

    cv::Mat color_slam = color.clone();
    cv::Mat depth_slam = depth.clone();
    cv::Mat cTw_orb = SLAM.TrackRGBD(color_slam, depth_slam, tframe);
    if (cTw_orb.empty()) continue;

    cTw = MatTofloat4x4(cTw_orb);
    wTc = cTw.getInverse();
    sensor.set_transform(wTc);

    map.Integrate(sensor);
    map.MarchingCubes();

    if (args.ray_casting) {
      ray_caster.Cast(map, cTw);
      cv::imshow("RayCasting", ray_caster.normal_image());
      cv::waitKey(1);
    }

    if (! args.mesh_range) {
      map.CollectAllBlocks();
    }
    map.CompressMesh();
    mesh.SetData(map.compact_mesh().vertices(),
                 (size_t)map.compact_mesh().vertex_count(),
                 map.compact_mesh().normals(),
                 (size_t)map.compact_mesh().vertex_count(),
                 NULL, 0,
                 map.compact_mesh().triangles(),
                 (size_t)map.compact_mesh().triangle_count());
    mesh_renderer.Render(cTw);

    if (args.record_video) {
      mesh_renderer.ScreenCapture(screen.data, screen.cols, screen.rows);
      cv::flip(screen, screen, 0);
      writer << screen;
    }
  }

  if (args.save_mesh) {
    map.SaveMesh(args.filename_prefix + ".obj");
  }

  SLAM.Shutdown();
  return 0;
}