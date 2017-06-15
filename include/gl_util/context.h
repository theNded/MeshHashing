//
// Created by Neo on 16/7/17.
// GLFW and GLEW initialization
// Provides @window() as an interaction interface
//

#ifndef GLUTILS_CONTEXT_H
#define GLUTILS_CONTEXT_H

#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace gl_utils {
class Context {
public:
  /// GLFWwindow* should exist on stack instead of heap
  Context();
  ~Context();

  int Init(size_t width, size_t height, std::string window_name,
           float z_near = kDefaultZnear, float z_far = kDefaultZfar);

  GLFWwindow *window() const {
    return window_;
  }
  const size_t width() const {
    return width_;
  }
  const size_t height() const {
    return height_;
  }
  const glm::mat4 projection_mat() {
    return projection_mat_;
  }

  // this version of glm is degree-based
  static const float kDefaultFov;
  static const float kDefaultZnear;
  static const float kDefaultZfar;

private:
  GLFWwindow *window_;
  size_t width_;
  size_t height_;

  glm::mat4 projection_mat_;
};
}


#endif //RAYTRACING_GLUTILS_CONTEXT_H
