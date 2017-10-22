//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_VISUALIZING_ENGINE_H
#define MESH_HASHING_VISUALIZING_ENGINE_H

#include <string>
#include "glwrapper.h"
#include "visualization/compact_mesh.h"

// TODO: setup a factory
class VisualizingEngine {
public:
  VisualizingEngine() = default;
  void Init(std::string, int width, int height);

  VisualizingEngine(std::string window_name, int width, int height);
  void set_interaction_mode(bool is_free);
  void UpdateViewpoint(glm::mat4 view);

  void SetMultiLightGeometryProgram(uint max_vertices,
                                    uint max_triangles,
                                    uint light_sources);
  void UpdateMultiLightGeometryUniforms(std::vector<glm::vec3> &light_src_positions,
                                        glm::vec3 light_color,
                                        float light_power);
  void UpdateMultiLightGeometryData(CompactMesh& compact_mesh);
  void RenderMultiLightGeometry(std::vector<glm::vec3> &light_src_positions,
                                glm::vec3 light_color,
                                float light_power,
                                CompactMesh &compact_mesh);

private:
  bool interaction_enabled_;

  // Shared viewpoint
  glm::mat4  mvp_;
  glm::mat4  view_;
  gl::Window window_;
  gl::Camera camera_;

  // Main shader
  // Problem with initialization if args is put below
  gl::Program  program_;
  gl::Uniforms uniforms_;
  gl::Args     args_;
};


#endif //MESH_HASHING_VISUALIZING_ENGINE_H
