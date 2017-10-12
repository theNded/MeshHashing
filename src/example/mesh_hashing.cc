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
#include "../sensor.h"
#include "../ray_caster.h"
#include <timer.h>
#include <queue>
#include "../tool/cpp/debugger.h"

#include "../dataset_manager.h"
#include "../opengl/window.h"
#include "../opengl/program.h"
#include "../opengl/camera.h"
#include "../opengl/uniforms.h"
#include "../opengl/args.h"


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

  gl::Window window("Mesh", config.sensor_params.width * 2, config.sensor_params.height * 2);
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

  map.use_fine_gradient()   = args.fine_gradient;

  cv::VideoWriter writer;
  cv::Mat screen;
  if (args.record_video) {
    writer = cv::VideoWriter("../result/3dv/" + args.filename_prefix + ".avi",
                             CV_FOURCC('X','V','I','D'),
                             30, cv::Size(config.sensor_params.width * 2,
                                          config.sensor_params.height * 2));
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

  //std::ofstream out_trs("trs.txt"), out_vtx_our("our.txt"), out_vtx_base("baseline.txt");
  std::ofstream time_prof("reduction.txt");
  double all_seconds = 0, meshing_seconds = 0, rendering_seconds = 0, compressing_seconds = 0;
  float3 prev_cam_pos;
  while (rgbd_data.ProvideData(depth, color, wTc)) {
    Timer timer_all, timer_meshing, timer_rendering, timer_compressing;

    frame_count ++;
    if (args.run_frames > 0 &&  frame_count > args.run_frames)
      break;


    sensor.Process(depth, color);
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();

    float3 camera_pos = make_float3(wTc.m14, wTc.m24, wTc.m34);
    float scale = 0.25;
    float4 v04 = wTc * make_float4(scale, scale, 2*scale, 1);
    float4 v14 = wTc * make_float4(scale, -scale, 2*scale, 1);
    float4 v24 = wTc * make_float4(-scale, scale, 2*scale, 1);
    float4 v34 = wTc * make_float4(-scale, -scale, 2*scale, 1);
    float3 v0 = make_float3(v04.x, v04.y, v04.z);
    float3 v1 = make_float3(v14.x, v14.y, v14.z);
    float3 v2 = make_float3(v24.x, v24.y, v24.z);
    float3 v3 = make_float3(v34.x, v34.y, v34.z);

    std::vector<float3> vs = {camera_pos, v0, camera_pos, v1, camera_pos, v2, camera_pos, v3,
                              v0, v1, v1, v3, v3, v2, v2, v0};

    prev_cam_pos = camera_pos;

    timer_all.Tick();
    map.Integrate(sensor);

    if (args.ray_casting) {
      ray_caster.Cast(map, cTw);
      cv::imshow("RayCasting", ray_caster.surface_image());
      cv::waitKey(1);
    }
//
//    if (frame_count > 1) // Re-estimate the SDF field
//      map.PlaneFitting(camera_pos);

    timer_meshing.Tick();
    map.MarchingCubes();


    if (! args.mesh_range) {
      map.CollectAllBlocks();
    }
    map.GetBoundingBoxes();
    double meshing_period = timer_meshing.Tock();
    meshing_seconds += meshing_period;

    timer_compressing.Tick();
    int3 stats;
    map.CompressMesh(stats);
    compressing_seconds += timer_compressing.Tock();

    timer_rendering.Tick();

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

    // If render mesh only:
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

//    glUseProgram(bbox_program.id());
//    glm::vec3 col = glm::vec3(1, 0, 0);
//    bbox_uniforms.Bind("mvp", &mvp);
//    bbox_uniforms.Bind("uni_color", &col);
//    traj_args.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
//                         vs.size(), vs.data());
//    glEnable(GL_LINE_SMOOTH);
//    glLineWidth(5.0f);
//    glDrawArrays(GL_LINES, 0, vs.size());


    window.swap_buffer();
    glfwPollEvents();

    if (window.get_key(GLFW_KEY_ESCAPE) == GLFW_PRESS ) {
      exit(0);
    }

    rendering_seconds += timer_rendering.Tock();
    all_seconds += timer_all.Tock();
    LOG(INFO) << frame_count / all_seconds;

    if (args.record_video) {
      cv::Mat rgb = window.CaptureRGB();
      cv::imshow("cap", rgb);
      writer << rgb;
    }

    time_prof << "(" << frame_count << ", " << 1000 * meshing_period << ")\n";

//    out_trs << "(" << frame_count << ", " << stats.x << ")\n";
//    out_vtx_our << "(" << frame_count << ", " << stats.y << ")\n";
//    out_vtx_base << "(" << frame_count << ", " << stats.z << ")\n";
  }

#ifdef DEBUG
//  debugger.CoreDump(map.compact_hash_table().gpu_data());
//  debugger.CoreDump(map.blocks().gpu_data());
//  debugger.CoreDump(map.mesh().gpu_data());
//  debugger.DebugAll();
#endif
  if (args.save_mesh) {
    map.SaveMesh("../result/3dv/" + args.filename_prefix + ".obj");
  }

  LOG(INFO) << (all_seconds - compressing_seconds)/ frame_count << "/" << all_seconds / frame_count;
  LOG(INFO) << meshing_seconds / frame_count;
  LOG(INFO) << rendering_seconds / frame_count;
  return 0;
}