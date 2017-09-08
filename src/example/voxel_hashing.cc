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
#include <queue>
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
                    config.sensor_params.width ,
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

  LineObject* traj;
  traj = new LineObject(30000);
  renderer.AddObject(traj);
  float3* traj_cuda;
  checkCudaErrors(cudaMalloc(&traj_cuda, 30000 * sizeof(float3)));

  renderer.free_walk() = args.free_walk;

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
                             30, cv::Size(config.sensor_params.width,
                                          config.sensor_params.height ));
    screen = cv::Mat(config.sensor_params.height ,
                     config.sensor_params.width ,
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
    LOG(INFO) << "Camera position: " << camera_pos.x << " " << camera_pos.y << " " << camera_pos.z;

    if (frame_count > 1) {
//      checkCudaErrors(cudaMemcpy(traj_cuda + 2 * frame_count - 2, &prev_cam_pos, sizeof(float3), cudaMemcpyHostToDevice));
//      checkCudaErrors(cudaMemcpy(traj_cuda + 2 * frame_count - 1, &camera_pos, sizeof(float3), cudaMemcpyHostToDevice));

      float scale = 0.25;
      float4 v1 = wTc * make_float4(scale, scale, 2*scale, 1);
      checkCudaErrors(cudaMemcpy(traj_cuda +  + 0, &camera_pos, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +  + 1, &v1, sizeof(float3), cudaMemcpyHostToDevice));

      float4 v2 = wTc * make_float4(scale, -scale, 2*scale, 1);
      checkCudaErrors(cudaMemcpy(traj_cuda +  + 2, &camera_pos, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +  + 3, &v2, sizeof(float3), cudaMemcpyHostToDevice));

      float4 v3 = wTc * make_float4(-scale, scale, 2*scale, 1) ;
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 4, &camera_pos, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 5, &v3, sizeof(float3), cudaMemcpyHostToDevice));

      float4 v4 = wTc * make_float4(-scale, -scale, 2*scale, 1);
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 6, &camera_pos, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 7, &v4, sizeof(float3), cudaMemcpyHostToDevice));

      checkCudaErrors(cudaMemcpy(traj_cuda +   + 8, &v1, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 9, &v2, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 10, &v2, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 11, &v4, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +  + 12, &v4, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 13, &v3, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 14, &v3, sizeof(float3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(traj_cuda +   + 15, &v1, sizeof(float3), cudaMemcpyHostToDevice));
      //traj->SetData(traj_cuda,  + 16, make_float3(0, 0, 1));
    }

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

    if (args.bounding_box) {
      bbox->SetData(map.bbox().vertices(),
                    map.bbox().vertex_count(), make_float3(1, 0, 0));
    }

    timer_rendering.Tick();
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
    rendering_seconds += timer_rendering.Tock();

    all_seconds += timer_all.Tock();
    LOG(INFO) << frame_count / all_seconds;

    if (args.record_video) {
      renderer.ScreenCapture(screen.data, screen.cols, screen.rows);
      cv::flip(screen, screen, 0);
      writer << screen;
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