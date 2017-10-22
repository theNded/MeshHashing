//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_VISUALIZING_ENGINE_H
#define MESH_HASHING_VISUALIZING_ENGINE_H

#include <string>
#include "glwrapper.h"
#include "visualization/compact_mesh.h"

class VisualizingEngine {
public:
  VisualizingEngine(std::string window_name, int width, int height);
  void set_interaction_mode(bool is_free);

  void SetMultiLightGeometryProgram(int max_vertices,
                                    int max_triangles,
                                    int light_sources);
  void UpdateViewpoint(glm::mat4 view);
  void UpdateMultiLightGeometryUniforms(std::vector<glm::vec3> &light_src_positions,
                                        glm::vec3 light_color,
                                        float light_power);
  void UpdateMultiLightGeometryData(CompactMesh& compact_mesh);
  void Render(std::vector<glm::vec3> &light_src_positions,
              glm::vec3 light_color,
              float light_power,
              CompactMesh& compact_mesh);

public:
  bool interaction_enabled_;

  glm::mat4  mvp_;
  glm::mat4  view_;
  gl::Window window_;
  gl::Camera camera_;

  // Main shader
  gl::Program  program_;
  gl::Uniforms uniforms_;
  gl::Args     args_;
};


#endif //MESH_HASHING_VISUALIZING_ENGINE_H
