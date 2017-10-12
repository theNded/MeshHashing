//
// Created by Neo on 14/08/2017.
//

#include "window.h"
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

namespace gl {
Window::Window(std::string window_name, int width, int height) {
  // Initialise GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW." << std::endl;
    exit(1);
  }
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Open a window and create its OpenGL context
  int res_factor = 1;
#ifdef __APPLE__
  // Retina requires 2x
  res_factor = 2;
#endif

  width_ = width / res_factor;
  height_ = height / res_factor;
  window_ = glfwCreateWindow(width_, height_, window_name.c_str(), NULL, NULL);
  if (window_ == NULL) {
    std::cerr << "Failed to open GLFW window." << std::endl;
    glfwTerminate();
    exit(1);
  }
  glfwMakeContextCurrent(window_);

  // Ensure we can capture the escape key being pressed below
  glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);

  // Initialize GLEW
  glewExperimental = GL_TRUE; // Needed for core profile
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW." << std::endl;
    glfwTerminate();
    exit(1);
  }

  img_width_  = width;
  img_height_ = height;
  rgb_   = cv::Mat(img_height_, img_width_, CV_8UC3);
  rgba_  = cv::Mat(img_height_, img_width_, CV_8UC4);
  depth_ = cv::Mat(img_height_, img_width_, CV_32F);
}

void Window::Resize(int width, int height) {
  int res_factor = 1;
#ifdef __APPLE__
  res_factor = 2;
#endif

  width_ = width / res_factor;
  height_ = height / res_factor;
  glfwSetWindowSize(window_, width_, height_);

  img_width_  = width;
  img_height_ = height;
  rgb_   = cv::Mat(img_height_, img_width_, CV_8UC3);
  rgba_  = cv::Mat(img_height_, img_width_, CV_8UC4);
  depth_ = cv::Mat(img_height_, img_width_, CV_32F);
}

cv::Mat Window::CaptureRGB() {
  glReadPixels(0, 0, img_width_, img_height_,
               GL_BGR, GL_UNSIGNED_BYTE, rgb_.data);
  cv::Mat ret;
  cv::flip(rgb_, ret, 0);

  return ret;
}

cv::Mat Window::CaptureRGBA() {
  glReadPixels(0, 0, img_width_, img_height_,
               GL_BGRA, GL_UNSIGNED_BYTE, rgba_.data);
  cv::Mat ret;
  cv::flip(rgba_, ret, 0);

  return ret;
}

cv::Mat Window::CaptureDepth() {
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, img_width_, img_height_,
               GL_DEPTH_COMPONENT, GL_FLOAT, depth_.data);
  cv::Mat ret;
  cv::flip(depth_, ret, 0);

  return ret;
}
}