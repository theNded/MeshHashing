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

#include "sensor/rgbd_local_sequence.h"
#include "sensor/rgbd_sensor.h"
#include "visualization/ray_caster.h"

#include "io/config_manager.h"
#include "core/collect.h"
#include "glwrapper.h"

#define DEBUG_

int main(int argc, char** argv) {
  std::vector<glm::vec3> light_src_positions = {
      glm::vec3(0, -2, 0),
      glm::vec3(4, -2, 0)
  };
  std::vector<glm::vec3> light_src_colors = {
      glm::vec3(1, 1, 1),
      glm::vec3(1, 1, 1)
  };
  float light_power = 3;
  glm::vec3 light_color = glm::vec3(1, 1, 1);
  std::stringstream ss;
  ss << light_src_positions.size();

  /// Use this to substitute tedious argv parsing
  RuntimeParams args;
  LoadRuntimeParams("../config/args.yml", args);

  ConfigManager config;
  RGBDLocalSequence rgbd_local_sequence;

  DatasetType dataset_type = DatasetType(args.dataset_type);
  config.LoadConfig(dataset_type);
  rgbd_local_sequence.LoadDataset(dataset_type);

  MainEngine main_engine(config.hash_params, config.mesh_params, config.sdf_params);
  // Add SetLights for main_engine
  VisualizingEngine vis_engine("Mesh", config.sensor_params.width*2, config.sensor_params.height*2);
  vis_engine.set_interaction_mode(args.free_walk);
  vis_engine.SetMultiLightGeometryProgram(config.mesh_params.max_vertex_count,
                                          config.mesh_params.max_triangle_count,
                                          light_src_positions.size());

//  gl::Program bbox_program;
//  bbox_program.Load(kShaderPath + "/line_vertex.glsl", gl::kVertexShader);
//  bbox_program.Load(kShaderPath + "/line_fragment.glsl", gl::kFragmentShader);
//  bbox_program.Build();
//
//  gl::Args bbox_args(1, true);
//  bbox_args.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
//                       config.hash_params.value_capacity * 24);
//  gl::Uniforms bbox_uniforms;
//  bbox_uniforms.GetLocation(bbox_program.id(), "mvp", gl::kMatrix4f);
//  bbox_uniforms.GetLocation(bbox_program.id(), "uni_color", gl::kVector3f);
//
//  gl::Args traj_args(1);
//  traj_args.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
//                       30000);

  Sensor    sensor(config.sensor_params);
  RayCaster ray_caster(config.ray_caster_params);

  main_engine.use_fine_gradient()   = args.fine_gradient;

  cv::VideoWriter writer;
  cv::Mat screen;
  if (args.record_video) {
    writer = cv::VideoWriter("../result/" + args.filename_prefix + ".avi",
                             CV_FOURCC('X','V','I','D'),
                             30, cv::Size(config.sensor_params.width * 2,
                                          config.sensor_params.height * 2));
  }

  cv::Mat color, depth;
  float4x4 wTc, cTw;
  int frame_count = 0;

  std::chrono::time_point<std::chrono::system_clock> start, end;

  double all_seconds = 0, meshing_seconds = 0, rendering_seconds = 0, compressing_seconds = 0;
  //float3 prev_cam_pos;
  while (rgbd_local_sequence.ProvideData(depth, color, wTc)) {
    Timer timer_all, timer_meshing, timer_rendering, timer_compressing;

    frame_count ++;
    if (args.run_frames > 0 &&  frame_count > args.run_frames)
      break;

    sensor.Process(depth, color);
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();
//
//    float3 camera_pos = make_float3(wTc.m14, wTc.m24, wTc.m34);
//    float scale = 0.25;
//    float4 v04 = wTc * make_float4(scale, scale, 2*scale, 1);
//    float4 v14 = wTc * make_float4(scale, -scale, 2*scale, 1);
//    float4 v24 = wTc * make_float4(-scale, scale, 2*scale, 1);
//    float4 v34 = wTc * make_float4(-scale, -scale, 2*scale, 1);
//    float3 v0 = make_float3(v04.x, v04.y, v04.z);
//    float3 v1 = make_float3(v14.x, v14.y, v14.z);
//    float3 v2 = make_float3(v24.x, v24.y, v24.z);
//    float3 v3 = make_float3(v34.x, v34.y, v34.z);
//
//    std::vector<float3> vs = {camera_pos, v0, camera_pos, v1, camera_pos, v2, camera_pos, v3,
//                              v0, v1, v1, v3, v3, v2, v2, v0};
//
//    prev_cam_pos = camera_pos;

    timer_all.Tick();
    main_engine.Mapping(sensor);
    main_engine.Meshing();
    main_engine.Recycle();
    //main_engine.Visualizing();
    // in main_engine.Visualizing():
    // if (ray_casting)
    //   ray_caster.Cast
    // if (! partial_mesh)
    //   CollectAll
    // CompressMesh
    // visualize_engine.Render()

    /////////////////////////////
    /// Should be in VisualizingEngine
    if (args.ray_casting) {
      ray_caster.Cast(main_engine.hash_table(), main_engine.blocks(), main_engine.converter(), cTw);
      cv::imshow("RayCasting", ray_caster.surface_image());
      cv::waitKey(1);
    }
//
//    if (frame_count > 1) // Re-estimate the SDF field
//      main_engine.PlaneFitting(camera_pos);

    timer_meshing.Tick();


    // TODO: add flag to blocks to deal with boundary conditions
    if (! args.mesh_range) {
      CollectAllBlockArray(main_engine.candidate_entries(), main_engine.hash_table());
    }

    main_engine.GetBoundingBoxes();
    double meshing_period = timer_meshing.Tock();
    meshing_seconds += meshing_period;

    timer_compressing.Tick();
    int3 stats;
    CompressMesh(main_engine.candidate_entries(),
                 main_engine.blocks(),
                 main_engine.mesh(),
                 main_engine.compact_mesh(),stats);
    compressing_seconds += timer_compressing.Tock();

    timer_rendering.Tick();

    /// Set uniform data
    glm::mat4 view;
    cTw = cTw.getTranspose();
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        view[i][j] = cTw.entries2[i][j];
    vis_engine.UpdateViewpoint(view);
    vis_engine.RenderMultiLightGeometry(light_src_positions,
                                        light_color,
                                        light_power,
                                        main_engine.compact_mesh());
/////////////////////////////

//    if (args.bounding_box) {
//      glUseProgram(bbox_program.id());
//      glm::vec3 col = glm::vec3(1, 0, 0);
//      bbox_uniforms.Bind("mvp", &mvp, 1);
//      bbox_uniforms.Bind("uni_color", &col, 1);
//
//      bbox_args.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
//                           main_engine.bbox().vertex_count(), main_engine.bbox().vertices());
//
//      glEnable(GL_LINE_SMOOTH);
//      glLineWidth(5.0f);
//      glDrawArrays(GL_LINES, 0, main_engine.bbox().vertex_count());
//    }

//    glUseProgram(bbox_program.id());
//    glm::vec3 col = glm::vec3(1, 0, 0);
//    bbox_uniforms.Bind("mvp", &mvp);
//    bbox_uniforms.Bind("uni_color", &col);
//    traj_args.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
//                         vs.size(), vs.data());
//    glEnable(GL_LINE_SMOOTH);
//    glLineWidth(5.0f);
//    glDrawArrays(GL_LINES, 0, vs.size());

    rendering_seconds += timer_rendering.Tock();
    all_seconds += timer_all.Tock();
    LOG(INFO) << frame_count / all_seconds;

//    if (args.record_video) {
//      cv::Mat rgb = window.CaptureRGB();
//      writer << rgb;
//    }


//    out_trs << "(" << frame_count << ", " << stats.x << ")\n";
//    out_vtx_our << "(" << frame_count << ", " << stats.y << ")\n";
//    out_vtx_base << "(" << frame_count << ", " << stats.z << ")\n";
  }

#ifdef DEBUG
//  debugger.CoreDump(main_engine.candidate_entries().gpu_memory());
//  debugger.CoreDump(main_engine.blocks().gpu_memory());
//  debugger.CoreDump(main_engine.meshing().gpu_memory());
//  debugger.DebugAll();
#endif
  if (args.save_mesh) {
    SavePly(main_engine.compact_mesh(), "../result/" + args.filename_prefix + ".ply");
  }

  LOG(INFO) << (all_seconds - compressing_seconds)/ frame_count << "/" << all_seconds / frame_count;
  LOG(INFO) << meshing_seconds / frame_count;
  LOG(INFO) << rendering_seconds / frame_count;
  return 0;
}