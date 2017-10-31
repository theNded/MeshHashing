//
// Created by wqy on 17-10-30.
//

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <chrono>

#include <engine/visualizing_engine.h>
#include <meshing/marching_cubes.h>
#include <visualization/color_util.h>

#include "sensor/rgbd_data_provider.h"

int main(int argc, char **argv) {
  LoggingEngine log_engine_;
  log_engine_.Init(".");

  int selected_frame_idx = 4;
  float truncation = 0.01f;
  BlockMap blocks;
  std::stringstream ss("");
  ss << selected_frame_idx;
  LOG(INFO) << "Reading block from disk";
  blocks = log_engine_.ReadRawBlocks(ss.str());

  std::vector<float3> pos;
  std::vector<float3> sdf;
  pos.reserve(blocks.size() * BLOCK_SIZE);
  sdf.reserve(blocks.size() * BLOCK_SIZE);
  for (auto &&block:blocks) {
    float delta = 1.0f / 2.0f;
    for (int i = 0; i < BLOCK_SIDE_LENGTH; ++i) {
      for (int j = 0; j < BLOCK_SIDE_LENGTH; ++j) {
        for (int k = 0; k < BLOCK_SIDE_LENGTH; ++k) {
          int index = i * 8 * 8 + j * 8 + k;
          float3 voxel_pos = {block.first.x + delta * i,
                              block.first.y + delta * j,
                              block.first.z + delta * k};
          pos.emplace_back(voxel_pos);
          sdf.emplace_back(ValToRGB(block.second.voxels[index].sdf,
                                    -truncation,
                                    truncation));
        }
      }
    }
  }
  LOG(INFO) << "Block info loaded";

  gl::Window window("BlockAnalysis", 640, 480);
  gl::Camera camera(window.visual_width(), window.visual_height());
  camera.SwitchInteraction(true);
  glm::mat4 model = glm::mat4(1.0);
  model[1][1] = -1;
  model[2][2] = -1;
  camera.set_model(model);
  gl::Program program;
  program.Load("../src/extern/opengl-wrapper/shader/block_analysis_vertex.glsl",
               gl::kVertexShader);
  program.Load("../src/extern/opengl-wrapper/shader/block_analysis_fragment.glsl",
               gl::kFragmentShader);
  program.Build();
  gl::Uniforms uniforms;
  uniforms.GetLocation(program.id(), "mvp", gl::kMatrix4f);

  gl::Args args(2);
  args.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                    pos.size(), pos.data());
  args.BindBuffer(1, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                    sdf.size(), sdf.data());

  glfwPollEvents();
  glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  do {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    camera.UpdateView(window);
    glm::mat4 mvp = camera.mvp();

    glUseProgram(program.id());
    uniforms.Bind("mvp", &mvp, 1);
    glBindVertexArray(args.vao());
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, pos.size());
    glBindVertexArray(0);

    window.swap_buffer();
  } while (window.get_key(GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           window.should_close() == 0);

  glfwTerminate();
  return 0;
}