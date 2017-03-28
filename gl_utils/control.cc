//
// Created by Neo on 16/7/17.
//

#include "control.h"

#include <cmath>
#include <iostream>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace gl_utils {
Control::Control(gl_utils::Context* context) {
  window_ = context->window();
  width_  = context->width();
  height_ = context->height();
  InitParameters();
}

Control::Control(GLFWwindow *window, int width, int height) {
  window_ = window;
  width_  = width;
  height_ = height;
  InitParameters();
}

glm::mat4 Control::view_mat() {
  return view_mat_;
}

glm::mat4 Control::projection_mat() {
  return projection_mat_;
}

void Control::UpdateCameraPose() {
  // glfwGetTime is called only once, the first time this function is called
  static double last_time = glfwGetTime();
  double current_time     = glfwGetTime();
  float delta_time        = float(current_time - last_time);

  // Get mouse position
#define ROTATION_FROM_KBD
#ifdef ROTATION_FROM_MOUSE
  double xpos, ypos;
  glfwGetCursorPos(window_, &xpos, &ypos);
  glfwSetCursorPos(window_, width_ / 2, height_ / 2);

  // Compute new orientation
  horizontal_angle_ -= rotate_speed_ * float(width_  / 2 - xpos);
  vertical_angle_   += rotate_speed_ * float(height_ / 2 - ypos);
#elif defined ROTATION_FROM_KBD
  if (glfwGetKey(window_, GLFW_KEY_I) == GLFW_PRESS) {
    vertical_angle_ += 0.01f;
  }
  if (glfwGetKey(window_, GLFW_KEY_K) == GLFW_PRESS) {
    vertical_angle_ -= 0.01f;
  }
  if (glfwGetKey(window_, GLFW_KEY_J) == GLFW_PRESS) {
    horizontal_angle_ -= 0.01f;
  }
  if (glfwGetKey(window_, GLFW_KEY_L) == GLFW_PRESS) {
    horizontal_angle_ += 0.01f;
  }
#endif

  glm::vec3 look_direction(
      cos(vertical_angle_) * sin(horizontal_angle_),
      sin(vertical_angle_),
      cos(vertical_angle_) * cos(horizontal_angle_));

  glm::vec3 move_direction(
      cos(vertical_angle_) * sin(horizontal_angle_),
      0,
      cos(vertical_angle_) * cos(horizontal_angle_));

  glm::vec3 right = glm::vec3(
      sin(horizontal_angle_ - M_PI_2),
      0,
      cos(horizontal_angle_ - M_PI_2));

  glm::vec3 up = glm::cross(right, look_direction);

  if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) {
    position_ -= look_direction * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) {
    position_ += look_direction * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) {
    position_ += right * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) {
    position_ -= right * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS) {
    position_ -= up * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
    position_ += up * move_speed_ * delta_time;
  }

  projection_mat_ = glm::perspective(fov_, (float)width_ / (float)height_,
                                     0.1f, 100.0f);
  // Camera matrix
  view_mat_ = glm::lookAt(
      position_,
      position_ + look_direction,
      up);

  last_time = current_time;
}

void Control::InitParameters() {
  position_         = glm::vec3(0, 0, 0);
  horizontal_angle_ = (float)M_PI;
  vertical_angle_   = 0.0f;
  // In later glm versions, fov_ = glm::radians(45.0f)
  fov_              = 45.0f;
  move_speed_       = 1.0f;
  rotate_speed_     = 0.005f;
}
}