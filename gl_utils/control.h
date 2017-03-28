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
  Control(gl_utils::Context *context);
  Control(GLFWwindow *window, int width, int height);
  void UpdateCameraPose();

  glm::mat4 view_mat();
  glm::mat4 projection_mat();

private:
  void InitParameters();

  // Descartes-system
  glm::mat4 view_mat_;
  glm::mat4 projection_mat_;
  glm::vec3 position_;

  // Polar-system parameters
  float horizontal_angle_;
  float vertical_angle_;
  float fov_;

  // Interaction parameters
  float rotate_speed_;
  float move_speed_;

  int width_;
  int height_;
  GLFWwindow *window_;
};
}

#endif //RAYTRACING_CONTROL_H
