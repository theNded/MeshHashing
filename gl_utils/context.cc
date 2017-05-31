//
// Created by Neo on 16/7/17.
//

#include "context.h"

#include <iostream>

#include <glog/logging.h>
#include <glm/gtc/matrix_transform.hpp>

namespace gl_utils {

/// !!! In later glm versions, fov_ = glm::radians(45.0f)
const float Context::kDefaultFov   = 45.0f;
const float Context::kDefaultZnear = 0.1f;
const float Context::kDefaultZfar  = 100.0f;

Context::Context() {
  width_ = height_ = 0;
  window_ = NULL;
}

Context::~Context() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

int Context::Init(size_t width, size_t height, std::string window_name,
                  float z_near, float z_far) {
  width_ = width;
  height_ = height;
  // Initialise GLFW
  if (!glfwInit()) {
    LOG(ERROR) << "Failed to initialize GLFW.";
    return -1;
  }

  LOG(INFO) << "Initializing GLFW window ...";
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Open a window and create its OpenGL context
  window_ = glfwCreateWindow(width_, height_, window_name.c_str(), NULL, NULL);
  if (window_ == NULL) {
    LOG(ERROR) << "Failed to open GLFW window.";
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window_);

  // Ensure we can capture the escape key being pressed below
  glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);
  glfwPollEvents();

  // Initialize GLEW
  glewExperimental = GL_TRUE; // Needed for core profile
  if (glewInit() != GLEW_OK) {
    LOG(ERROR) << "Failed to initialize GLEW.";
    glfwTerminate();
    return -1;
  }

  projection_mat_ = glm::perspective(kDefaultFov,
                                     (float)width_ / (float)height_,
                                     z_near, z_far);

  return 0;
}
}