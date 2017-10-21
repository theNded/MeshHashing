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

#include "../core/params.h"
#include "../io/dataset_manager.h"
#include "../engine/map.h"
#include "../engine/sensor.h"
#include "../visualization/ray_caster.h"
#include "../opengl/args.h"
#include "../opengl/uniforms.h"
#include "../opengl/program.h"
#include "../opengl/window.h"
#include "../opengl/camera.h"

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
  gl::Window window("Mesh", config.sensor_params.width, config.sensor_params.height);
  gl::Camera camera(window.width(), window.height());
  camera.SwitchInteraction(true);
  glm::mat4 p = camera.projection();
  glm::mat4 m = glm::mat4(1.0f);
  m[1][1] = -1;
  m[2][2] = -1;

  gl::Program program;
  gl::Uniforms uniforms;
  if (args.render_type == 0) {
    program.Build("../shader/mesh_vn_vertex.glsl",
                  "../shader/mesh_vn_fragment.glsl");
  } else {
    program.Build("../shader/mesh_vc_vertex.glsl",
                  "../shader/mesh_vc_fragment.glsl");
  }
  uniforms.GetLocation(program.id(), "mvp", gl::kMatrix4f);
  if (args.render_type == 0) {
    uniforms.GetLocation(program.id(), "view_mat", gl::kMatrix4f);
    uniforms.GetLocation(program.id(), "model_mat", gl::kMatrix4f);
  }

  gl::Args glargs(3, true);
  glargs.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                    config.mesh_params.max_vertex_count);
  glargs.InitBuffer(1, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                    config.mesh_params.max_vertex_count);
  glargs.InitBuffer(2, {GL_ARRAY_BUFFER, sizeof(int), 3, GL_INT},
                    config.mesh_params.max_triangle_count);

  gl::Program bbox_program("../shader/line_vertex.glsl",
                           "../shader/line_fragment.glsl");
  gl::Args bbox_args(1, true);
  bbox_args.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                       config.hash_params.value_capacity * 24);
  gl::Uniforms bbox_uniforms;
  bbox_uniforms.GetLocation(bbox_program.id(), "mvp", gl::kMatrix4f);
  bbox_uniforms.GetLocation(bbox_program.id(), "uni_color", gl::kVector3f);

  gl::Args traj_args(1);
  traj_args.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                       30000);

  SetConstantSDFParams(config.sdf_params);
  Map       map(config.hash_params, config.mesh_params,
                "../result/3dv/" + args.time_profile + ".txt",
                "../result/3dv/" + args.memo_profile + ".txt");
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
      map.CollectAllBlockArray();
    }
    int3 stats;
    map.CompressMesh(stats);


    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program.id());

    /// Set uniform data
    glm::mat4 view;
    cTw = cTw.getTranspose();
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        view[i][j] = cTw.entries2[i][j];
    view = m * view * glm::inverse(m);
    if (args.free_walk) {
      camera.SetView(window);
      view = camera.view();
    }
    glm::mat4 mvp = p * view * m;
    uniforms.Bind("mvp", &mvp);
    if (args.render_type == 0) {
      uniforms.Bind("view_mat", &view);
      uniforms.Bind("model_mat", &m);
    }

    /// Set args data
    if (args.render_type == 0) {
      glargs.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                        map.compact_mesh().vertex_count(), map.compact_mesh().vertices());
      glargs.BindBuffer(1, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                        map.compact_mesh().vertex_count(), map.compact_mesh().normals());
      glargs.BindBuffer(2, {GL_ELEMENT_ARRAY_BUFFER, sizeof(int), 3, GL_INT},
                        map.compact_mesh().triangle_count(), map.compact_mesh().triangles());
    } else {
      glargs.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                        map.compact_mesh().vertex_count(), map.compact_mesh().vertices());
      glargs.BindBuffer(1, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                        map.compact_mesh().vertex_count(), map.compact_mesh().colors());
      glargs.BindBuffer(2, {GL_ELEMENT_ARRAY_BUFFER, sizeof(int), 3, GL_INT},
                        map.compact_mesh().triangle_count(), map.compact_mesh().triangles());
    }

    // If render meshing only:
    if (args.ploygon_mode) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }

    /// NOTE: Use GL_UNSIGNED_INT instead of GL_INT, otherwise it won't work
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, map.compact_mesh().triangle_count() * 3, GL_UNSIGNED_INT, 0);

    if (args.bounding_box) {
      glUseProgram(bbox_program.id());
      glm::vec3 col = glm::vec3(1, 0, 0);
      bbox_uniforms.Bind("mvp", &mvp);
      bbox_uniforms.Bind("uni_color", &col);

      bbox_args.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                           map.bbox().vertex_count(), map.bbox().vertices());

      glEnable(GL_LINE_SMOOTH);
      glLineWidth(5.0f);
      glDrawArrays(GL_LINES, 0, map.bbox().vertex_count());
    }

    window.swap_buffer();
    glfwPollEvents();

    if (window.get_key(GLFW_KEY_ESCAPE) == GLFW_PRESS ) {
      exit(0);
    }
    if (args.record_video) {
      cv::Mat rgb = window.CaptureRGB();
      cv::flip(rgb, rgb, 0);
      writer << rgb;
    }
  }

  if (args.save_mesh) {
    map.SaveMesh(args.filename_prefix + ".obj");
  }

  SLAM.Shutdown();
  return 0;
}