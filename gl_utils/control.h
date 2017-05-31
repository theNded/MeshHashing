//
// Created by Neo on 16/7/17.
// Specify mouse and keyboard control
//

#ifndef GLUTILS_CONTROL_H
#define GLUTILS_CONTROL_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "context.h"

namespace gl_utils {
class Control {
public:
  Control(const Context& context);
  Control(GLFWwindow *window, size_t width, size_t height);
  void UpdateCameraPose();

  glm::mat4 view_mat() {
    return view_mat_;
  }

private:
  void InitParameters();

  // Descartes-system
  glm::mat4 view_mat_;
  glm::vec3 position_;

  // Polar-system parameters
  float horizontal_angle_;
  float vertical_angle_;

  // Interaction parameters
  float rotate_speed_;
  float move_speed_;

  GLFWwindow *window_;
  size_t width_;
  size_t height_;
};
}

#endif //RAYTRACING_CONTROL_H
