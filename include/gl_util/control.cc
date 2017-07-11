//
// Created by Neo on 16/7/17.
//

#include "control.h"

#include <cmath>
#include <iostream>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glog/logging.h>

namespace gl_utils {
Control::Control(const Context& context) {
  window_ = context.window();
  width_  = context.width();
  height_ = context.height();
  InitParameters();
}

Control::Control(GLFWwindow *window, size_t width, size_t height) {
  window_ = window;
  width_  = width;
  height_ = height;
  InitParameters();
}


void Control::UpdateCameraPose() {
  // glfwGetTime is called only once, the first time this function is called
  static double last_time = glfwGetTime();
  double current_time     = glfwGetTime();
  float delta_time        = float(current_time - last_time);

  // Get mouse position
#define ROTATION_FROM_KBD
//#define ROTATION_FROM_MOUSE
#ifdef ROTATION_FROM_MOUSE
  double xpos, ypos;
  glfwGetCursorPos(window_, &xpos, &ypos);
  glfwSetCursorPos(window_, width_ / 2, height_ / 2);

  // Compute new orientation
  horizontal_angle_ -= rotate_speed_ * float(width_  / 2 - xpos);
  vertical_angle_   += rotate_speed_ * float(height_ / 2 - ypos);
#elif defined ROTATION_FROM_KBD
  if (glfwGetKey(window_, GLFW_KEY_I) == GLFW_PRESS) {
    vertical_angle_ += rotate_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_K) == GLFW_PRESS) {
    vertical_angle_ -= rotate_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_J) == GLFW_PRESS) {
    horizontal_angle_ += rotate_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_L) == GLFW_PRESS) {
    horizontal_angle_ -= rotate_speed_ * delta_time;
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
    position_ += look_direction * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) {
    position_ -= look_direction * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) {
    position_ += right * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) {
    position_ -= right * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS) {
    position_ += up * move_speed_ * delta_time;
  }
  if (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
    position_ -= up * move_speed_ * delta_time;
  }

  LOG(INFO) << "----------";
  LOG(INFO) << position_.x << " " << position_.y << " " << position_.z;
  LOG(INFO) << horizontal_angle_;
  LOG(INFO) << vertical_angle_;

  // Camera matrix
  view_mat_ = glm::lookAt(
      position_,
      position_ + look_direction,
      up);

  last_time = current_time;
}

void Control::InitParameters() {
  // TUM3
//  position_ = glm::vec3(-0.0381861, 2.91101, -1.57104);
//  horizontal_angle_ = (float)M_PI;
//  vertical_angle_ = -1.123f;

  // lounge
  position_ = glm::vec3(0.272362, 0.877346, 1.46175);
  horizontal_angle_ = 2.942f;
  vertical_angle_ = -0.112f;


  // Default
  //position_         = glm::vec3(0, 2, 0);
  //horizontal_angle_ = (float)M_PI;
  //vertical_angle_   = -(float)M_PI_2; // 0.0
  move_speed_       = 0.5f;
  rotate_speed_     = 0.2f;
}
}