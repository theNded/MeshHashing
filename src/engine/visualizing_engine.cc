//
// Created by wei on 17-10-22.
//

#include <glog/logging.h>
#include "visualizing_engine.h"

const std::string kShaderPath = "../src/extern/opengl-wrapper/shader";

VisualizingEngine::VisualizingEngine(std::string window_name, int width, int height) {
  window_.Init(window_name, width, height);
  camera_.set_perspective(width, height);
  glm::mat4 m = glm::mat4(1.0f);
  m[1][1] = -1;
  m[2][2] = -1;
  camera_.set_model(m);
}

void VisualizingEngine::SetMultiLightGeometryProgram(int max_vertices,
                                                     int max_triangles,
                                                     int light_sources) {
  std::stringstream ss("");
  ss << light_sources;
  LOG(INFO) << ss.str();
  program_.Load(kShaderPath + "/model_multi_light_vertex.glsl", gl::kVertexShader);
  program_.ReplaceMacro("LIGHT_COUNT", ss.str(), gl::kVertexShader);
  program_.Load(kShaderPath + "/model_multi_light_fragment.glsl", gl::kFragmentShader);
  program_.ReplaceMacro("LIGHT_COUNT", ss.str(), gl::kFragmentShader);
  program_.Build();

  uniforms_.GetLocation(program_.id(), "mvp", gl::kMatrix4f);
  uniforms_.GetLocation(program_.id(), "c_T_w", gl::kMatrix4f);
  uniforms_.GetLocation(program_.id(), "light",gl::kVector3f);
  uniforms_.GetLocation(program_.id(), "light_power",gl::kFloat);
  uniforms_.GetLocation(program_.id(), "light_color",gl::kVector3f);

  args_.Init(3, true);
  args_.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                    max_vertices);
  args_.InitBuffer(1, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                    max_vertices);
  args_.InitBuffer(2, {GL_ELEMENT_ARRAY_BUFFER, sizeof(int), 3, GL_UNSIGNED_INT},
                    max_triangles);
};

void VisualizingEngine::UpdateViewpoint(glm::mat4 view) {
  view_ = camera_.model() * view * glm::inverse(camera_.model());
  mvp_ = camera_.projection() * view_ * camera_.model();
}

void VisualizingEngine::UpdateMultiLightGeometryUniforms(
    std::vector<glm::vec3> &light_src_positions,
    glm::vec3 light_color,
    float light_power) {
  uniforms_.Bind("mvp", &mvp_, 1);
  uniforms_.Bind("c_T_w", &view_, 1);
  uniforms_.Bind("light", light_src_positions.data(), light_src_positions.size());
  uniforms_.Bind("light_color", &light_color, 1);
  uniforms_.Bind("light_power", &light_power, 1);
}

void VisualizingEngine::UpdateMultiLightGeometryData(CompactMesh &compact_mesh) {
  LOG(INFO) << 0;
  LOG(INFO) << compact_mesh.vertex_count();
  args_.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                   compact_mesh.vertex_count(), compact_mesh.vertices());
  LOG(INFO) << 1;
  args_.BindBuffer(1, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                   compact_mesh.vertex_count(), compact_mesh.normals());
  LOG(INFO) << 2;
  args_.BindBuffer(2, {GL_ELEMENT_ARRAY_BUFFER, sizeof(int), 3, GL_UNSIGNED_INT},
                   compact_mesh.triangle_count(), compact_mesh.triangles());
}

void VisualizingEngine::set_interaction_mode(bool is_free) {
  camera_.SwitchInteraction(is_free);
}

void VisualizingEngine::Render(std::vector<glm::vec3> &light_src_positions,
                               glm::vec3 light_color,
                               float light_power,
                               CompactMesh& compact_mesh) {
  glClearColor(1, 1, 1, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(program_.id());

  LOG(INFO) << "uniforms";
  UpdateMultiLightGeometryUniforms(light_src_positions,
                                   light_color,
                                   light_power);
  LOG(INFO) << "data";
  UpdateMultiLightGeometryData(compact_mesh);

  LOG(INFO) << "render";
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  /// NOTE: Use GL_UNSIGNED_INT instead of GL_INT, otherwise it won't work
  glDrawElements(GL_TRIANGLES, compact_mesh.triangle_count() * 3, GL_UNSIGNED_INT, 0);

  window_.swap_buffer();
  if (window_.get_key(GLFW_KEY_ESCAPE) == GLFW_PRESS ) {
    exit(0);
  }
}