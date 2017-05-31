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

#define TUM3
#if defined(ICL)
const std::string kDefaultDatasetPath = "/home/wei/data/ICL/lv2/";
#elif defined(TUM1)
const std::string kDefaultDatasetPath =
        "/home/wei/data/TUM/rgbd_dataset_freiburg1_xyz/";
#elif defined(TUM3)
const std::string kDefaultDatasetPath =
        "/home/wei/data/TUM/rgbd_dataset_freiburg3_long_office_household/";
#elif defined(SUN3D)
const std::string kDefaultDatasetPath =
        "/home/wei/data/SUN3D/copyroom/";
#elif defined(SUN3D_ORI)
const std::string kDefaultDatasetPath =
        "/home/wei/data/SUN3D-Princeton/hotel_umd/maryland_hotel3/";
#elif defined(TDVCR)
const std::string kDefaultDatasetPath =
        "/home/wei/data/3DVCR/lab3/";
#endif

std::string path_to_vocabulary = "../../orb_slam2/Vocabulary/ORBvoc.bin";
std::string path_to_orb_config = "../config/ORB/TUM3.yaml";

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
  /// Load images
  std::vector<std::string> depth_img_list;
  std::vector<std::string> color_img_list;
  std::vector<float4x4>    wTcs;

  ConfigManager config;
#if defined(ICL)
  LoadICL(kDefaultDatasetPath, depth_img_list, color_img_list, wTcs);
  config.LoadConfig("../config/icl.yml");
#elif defined(TUM1)
  LoadTUM(kDefaultDatasetPath, depth_img_list, color_img_list, wTcs);
  config.LoadConfig("../config/tum1.yml");
#elif defined(TUM3)
  LoadTUM(kDefaultDatasetPath, depth_img_list, color_img_list, wTcs);
  config.LoadConfig("../config/TUM3.yml");
#elif defined(SUN3D)
  LoadSUN3D(kDefaultDatasetPath, depth_img_list, color_img_list, wTcs);
  config.LoadConfig("../config/sun3d.yml");
#elif defined(SUN3D_ORI)
  LoadSUN3DOriginal(kDefaultDatasetPath, depth_img_list, color_img_list, wTcs);
  config.LoadConfig("../config/sun3d_ori.yml");
#elif defined(TDVCR)
  Load3DVCR(kDefaultDatasetPath, depth_img_list, color_img_list, wTcs);
  config.LoadConfig("../config/3dvcr.yml");
#endif

  ORB_SLAM2::System SLAM(path_to_vocabulary, path_to_orb_config,
                         ORB_SLAM2::System::RGBD, true);

  MeshRenderer renderer("Mesh",
                        config.sensor_params.width,
                        config.sensor_params.height);
  renderer.free_walk() = false;

  std::vector<std::string> uniform_names;
  uniform_names.push_back("mvp");
  renderer.CompileShader("../shader/mesh_vertex.glsl",
                         "../shader/mesh_fragment.glsl",
                         uniform_names);
  SetConstantSDFParams(config.sdf_params);

  Map voxel_map(config.hash_params);
  LOG(INFO) << "Map allocated";

  Sensor sensor(config.sensor_params);
  sensor.BindGPUTexture();

  RayCaster ray_caster(config.ray_caster_params);

  // Main loop
  int frames = depth_img_list.size() - 1;

  for (int i = 0; i < 10; ++i) {
    LOG(INFO) << i;
    cv::Mat depth = cv::imread(depth_img_list[i], -1);
    cv::Mat color = cv::imread(color_img_list[i]);

    cv::cvtColor(color, color, CV_BGR2BGRA);
    sensor.Process(depth, color);

    cv::Mat color_slam = color.clone();
    cv::Mat depth_slam = depth.clone();
    double tframe;
    cv::Mat Tcw = SLAM.TrackRGBD(color_slam, depth_slam, tframe);
    if (Tcw.empty()) continue;

    float4x4 cTw = MatTofloat4x4(Tcw);
    float4x4 wTc = cTw.getInverse();
    sensor.set_transform(wTc);

    voxel_map.Integrate(sensor, NULL);
    voxel_map.MarchingCubes();
    voxel_map.CompressMesh();

    //ray_caster.Cast(voxel_map, cTw.getInverse());
    voxel_map.CollectAllBlocks();
    voxel_map.CompressMesh();
    renderer.Render(voxel_map.compact_mesh().vertices(),
                    (size_t)voxel_map.compact_mesh().vertex_count(),
                    voxel_map.compact_mesh().normals(),
                    (size_t)voxel_map.compact_mesh().vertex_count(),
                    voxel_map.compact_mesh().triangles(),
                    (size_t)voxel_map.compact_mesh().triangle_count(),
                    cTw);
    //cv::imshow("normal", ray_caster.normal_image());
    //cv::waitKey(1);
  }

  // Stop all threads
  SLAM.Shutdown();

  return 0;
}