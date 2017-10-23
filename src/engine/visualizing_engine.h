//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_VISUALIZING_ENGINE_H
#define MESH_HASHING_VISUALIZING_ENGINE_H

#include <string>
#include "glwrapper.h"
#include "visualization/compact_mesh.h"
#include "visualization/ray_caster.h"

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

  void set_interaction_mode(bool is_free);
  void set_light(Light& light);
  void update_view_matrix();
  void set_view_matrix(glm::mat4 view);

  // Call set_lights before this
  void BuildMultiLightGeometryProgram(uint max_vertices,
                                      uint max_triangles,
                                      bool enable_global_mesh);
  void BindMultiLightGeometryUniforms();
  void BindMultiLightGeometryData(CompactMesh &compact_mesh);
  void RenderMultiLightGeometry(CompactMesh &compact_mesh);

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
private:
  bool enable_interaction_;
  bool enable_global_mesh_;
  bool enable_ray_casting_;

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

  gl::Program  box_program_;
  gl::Uniforms box_uniforms_;

  // Raycaster
  RayCaster ray_caster_;
};


#endif //MESH_HASHING_VISUALIZING_ENGINE_H
