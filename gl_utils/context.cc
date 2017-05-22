//
// Created by Neo on 16/7/17.
//

#include "context.h"

#include <iostream>

#include <glog/logging.h>

namespace gl_utils {
Context::Context(std::string window_name, size_t width, size_t height) {
  width_ = width;
  height_ = height;
  Init(window_name);
}

int Context::Init(std::string window_name) {
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

  return 0;
}

GLFWwindow *Context::window() const {
  return window_;
}

size_t Context::width() {
  return width_;
}

size_t Context::height() {
  return height_;
}
}