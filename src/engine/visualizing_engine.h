//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_VISUALIZING_ENGINE_H
#define MESH_HASHING_VISUALIZING_ENGINE_H

#include <string>
#include <visualization/trajectory.h>
#include "glwrapper.h"
#include "visualization/compact_mesh.h"
#include "visualization/ray_caster.h"
#include "visualization/bounding_box.h"

struct Light {
  std::vector<glm::vec3> light_srcs;
  glm::vec3 light_color;
  float light_power;
};

// TODO: setup a factory
class VisualizingEngine {
public:
  VisualizingEngine() = default;
  void Init(std::string, int width, int height);
  VisualizingEngine(std::string window_name, int width, int height);
  ~VisualizingEngine();

  void set_interaction_mode(bool is_free);
  void update_view_matrix();
  void set_view_matrix(glm::mat4 view);

  void Render();
  cv::Mat Capture() {
    return window_.CaptureRGB();
  }
  // Call set_lights before this
  void set_light(Light& light);
  void BindMainProgram(uint max_vertices,
                       uint max_triangles,
                       bool enable_global_mesh);
  void BindMainUniforms();
  void BindMainData();
  void RenderMain();

  // At current assume all kinds of data uses the same program
  void BuildHelperProgram();
  void BindHelperUniforms();
  void RenderHelper();

  void InitBoundingBoxData(uint max_vertices);
  void BindBoundingBoxData();
  void InitTrajectoryData(uint max_vertices);
  void BindTrajectoryData();

  void BuildRayCaster(const RayCasterParams& ray_caster_params);
  void RenderRayCaster(float4x4 view, HashTable& hash_table, BlockArray& blocks, CoordinateConverter& converter) ;

  bool enable_interaction() {
    return enable_interaction_;
  }
  bool enable_global_mesh() {
    return enable_global_mesh_;
  }
  bool enable_ray_casting() {
    return enable_ray_casting_;
  }
  bool enable_bounding_box() {
    return enable_bounding_box_;
  }
  bool enable_trajectory() {
    return enable_trajectory_;
  }
  CompactMesh& compact_mesh() {
    return compact_mesh_;
  }
  BoundingBox& bounding_box() {
    return bounding_box_;
  }
  Trajectory& trajectory() {
    return trajectory_;
  }

private:
  bool enable_interaction_ = false;
  bool enable_global_mesh_ = false;
  bool enable_ray_casting_ = false;
  bool enable_bounding_box_ = false;
  bool enable_trajectory_   = false;

  // Lighting conditions
  Light light_;

  // Shared viewpoint
  glm::mat4  mvp_;
  glm::mat4  view_;
  gl::Window window_;
  gl::Camera camera_;

  // Main shader
  gl::Program  main_program_;
  gl::Uniforms main_uniforms_;
  gl::Args     main_args_;

  gl::Program  helper_program_;
  gl::Uniforms helper_uniforms_;

  gl::Args     box_args_;
  gl::Args     trajectory_args_;

  // Raycaster
  RayCaster   ray_caster_;
  CompactMesh compact_mesh_;
  BoundingBox bounding_box_;
  Trajectory  trajectory_;
};


#endif //MESH_HASHING_VISUALIZING_ENGINE_H
