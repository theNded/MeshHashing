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

#include "renderer.h"

#include "config_reader.h"
#include "control.h"

#define SUN3D
#if defined(ICL)
const std::string kDefaultDatasetPath = "/home/wei/data/ICL/lv2/";
#elif defined(TUM)
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
        "/home/wei/data/3DVCR/hall1/";
#endif

/// Refer to constant.cu
extern void SetConstantSDFParams(const SDFParams& params);

float4x4 perspective(
        const float & fovy,
        const float & aspect,
        const float & zNear,
        const float & zFar) {
  assert(zFar != zNear);

  float tanHalfFovy = tanf(fovy / 2);

  float4x4 m;
  m.setValue(0);
  m.entries2[0][0] = 1 / (aspect * tanHalfFovy);
  m.entries2[1][1] = 1 / (tanHalfFovy);
  m.entries2[2][2] = -(zFar + zNear) / (zFar - zNear);
  m.entries2[2][3] = -1;
  m.entries2[3][2] = -(2 * zFar * zNear) / (zFar - zNear);
  return m;
}

int main(int argc, char** argv) {
  /// Load images
  std::vector<std::string> depth_img_list;
  std::vector<std::string> color_img_list;
  std::vector<float4x4>    wTc;

  ConfigReader config;
#if defined(ICL)
  LoadICL(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/icl.yml");
#elif defined(TUM)
  LoadTUM(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/tum3.yml");
#elif defined(SUN3D)
  LoadSUN3D(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/sun3d.yml");
#elif defined(SUN3D_ORI)
  LoadSUN3DOriginal(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/sun3d_ori.yml");
#elif defined(TDVCR)
  Load3DVCR(kDefaultDatasetPath, depth_img_list, color_img_list, wTc);
  config.LoadConfig("../config/3dvcr.yml");
#endif

  RendererBase::InitGLWindow("Display",
                             config.ray_caster_params.width,
                             config.ray_caster_params.height);
  RendererBase::InitCUDA();
  SetConstantSDFParams(config.sdf_params);

  Map voxel_map(config.hash_params);
  LOG(INFO) << "Map allocated";

  Sensor sensor(config.sensor_params);
  sensor.BindGPUTexture();
  float4x4 K; K.setIdentity();
  K.m11 = config.sensor_params.fx;
  K.m13 = config.sensor_params.cx;
  K.m22 = config.sensor_params.fy;
  K.m23 = config.sensor_params.cy;

  RayCaster ray_caster(config.ray_caster_params);

  //cv::VideoWriter writer("icl-vh.avi", CV_FOURCC('X','V','I','D'),
  //                       30, cv::Size(640, 480));

  std::vector<std::string> uniform_names;

  FrameRenderer frame_renderer;
  uniform_names.clear();
  uniform_names.push_back("texture_sampler");
  frame_renderer.CompileShader("../shader/frame_vertex.glsl",
                               "../shader/frame_fragment.glsl",
                               uniform_names);

  MeshRenderer mesh_renderer;
  uniform_names.clear();
  uniform_names.push_back("mvp");
  mesh_renderer.CompileShader("../shader/mesh_vertex.glsl",
                              "../shader/mesh_fragment.glsl",
                              uniform_names);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int frames = depth_img_list.size() - 1;
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  gl_utils::Control control(RendererBase::window(), 640, 480);

  for (int i = 0; i < frames; ++i) {
    LOG(INFO) << i;
    cv::Mat depth = cv::imread(depth_img_list[i], -1);
    cv::Mat color = cv::imread(color_img_list[i]);

    cv::cvtColor(color, color, CV_BGR2BGRA);

    sensor.Process(depth, color);
    float4x4 T = wTc[0].getInverse() * wTc[i];
    sensor.set_transform(T);

    voxel_map.Integrate(sensor, NULL);
    voxel_map.MarchingCubes();

//

    //ray_caster.Cast(voxel_map, T.getInverse());

//    float4x4 transfer;
//    transfer.setIdentity();
//    transfer.m22 = -1;
//    transfer.m33 = -1;
//
//    float4x4 mvp = perspective(0.74f, 4.0f / 3.0f, 0.01f, 100.0f)
//                   * transfer * T.getInverse() * transfer;


    control.UpdateCameraPose();
    glm::mat4 transform = glm::mat4(0);
    transform[0][0] = 1;
    transform[1][1] = -1;
    transform[2][2] = -1;
    transform[3][3] = 1;

    glm::mat4 view_mat;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        view_mat[i][j] = T.entries2[i][j];

    glm::mat4 mvp = control.projection_mat() *
                    transform *
                    control.view_mat() *
                    glm::mat4(1.0f);
    glm::mat4 v = control.view_mat();
            //T.getInverse();
    //frame_renderer.Render(ray_caster.gpu_data().normal_image);
    LOG(INFO) << voxel_map.compact_mesh().vertex_count();

    voxel_map.CollectAllBlocks();
    voxel_map.CompressMesh();
    mesh_renderer.Render(voxel_map.compact_mesh().vertices(),
                         (size_t)voxel_map.compact_mesh().vertex_count(),
                         voxel_map.compact_mesh().triangles(),
                         (size_t)voxel_map.compact_mesh().triangle_count(),
                         &mvp[0][0]);
//    cv::imshow("display", ray_caster.normal_image());
//    cv::waitKey(1);
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  LOG(INFO) << "Total time: " << seconds.count();
  LOG(INFO) << "Fps: " << frames / seconds.count();

  voxel_map.SaveMesh("kkk.obj");
  RendererBase::DestroyGLWindow();
}