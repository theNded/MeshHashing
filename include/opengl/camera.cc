//
// Created by Neo on 14/08/2017.
//

#include <glm/gtc/matrix_transform.hpp>
#include "camera.h"

namespace gl {
Camera::Camera(int width, int height,
               float fov, float z_near, float z_far) {
  set_perspective(width, height, fov, z_near, z_far);
  view_  = glm::mat4(1.0f);
  model_ = glm::mat4(1.0f);
}

void Camera::SwitchInteraction(bool enable_interaction) {
  interaction_enabled_ = enable_interaction;
  // On initialization
  if (interaction_enabled_) {
    position_ = glm::vec3(0, 0, 0);
    azimuth_ = (float)M_PI;
    elevation_ = 0.0f;
  }
}

void Camera::SetView(Window &window) {
  if (! interaction_enabled_) return;

  static double last_time = glfwGetTime();
  double current_time = glfwGetTime();
  float delta_time = float(current_time - last_time);

  if (window.get_key(GLFW_KEY_UP)) {
    elevation_   += kRotateSpeed * delta_time;
  } else if (window.get_key(GLFW_KEY_DOWN)) {
    elevation_   -= kRotateSpeed * delta_time;
  } else if (window.get_key(GLFW_KEY_LEFT)) {
    azimuth_ += kRotateSpeed * delta_time;
  } else if (window.get_key(GLFW_KEY_RIGHT)) {
    azimuth_ -= kRotateSpeed * delta_time;
  }

  // Compute new orientation
  glm::vec3 look_direction(
      cos(elevation_) * sin(azimuth_),
      sin(elevation_),
      cos(elevation_) * cos(azimuth_));

  glm::vec3 move_direction(
      cos(elevation_) * sin(azimuth_),
      0,
      cos(elevation_) * cos(azimuth_));

  glm::vec3 right = glm::vec3(
      sin(azimuth_ - M_PI_2),
      0,
      cos(azimuth_ - M_PI_2));

  glm::vec3 up = glm::cross(right, look_direction);

  if (window.get_key(GLFW_KEY_W) == GLFW_PRESS) {
    position_ += move_direction * kMoveSpeed * delta_time;
  }
  if (window.get_key(GLFW_KEY_S) == GLFW_PRESS) {
    position_ -= move_direction * kMoveSpeed * delta_time;
  }
  if (window.get_key(GLFW_KEY_D) == GLFW_PRESS) {
    position_ += right * kMoveSpeed * delta_time;
  }
  if (window.get_key(GLFW_KEY_A) == GLFW_PRESS) {
    position_ -= right * kMoveSpeed * delta_time;
  }
  if (window.get_key(GLFW_KEY_SPACE) == GLFW_PRESS) {
    position_.y += kMoveSpeed * delta_time;
  }
  if (window.get_key(GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
    position_.y -= kMoveSpeed * delta_time;
  }

  // Camera matrix
  view_ = glm::lookAt(position_,
                      position_ + look_direction,
                      up);

  last_time = current_time;
}

cv::Mat Camera::ConvertDepthBuffer(cv::Mat& depthf, float factor) {
  cv::Mat depths = cv::Mat(depthf.rows, depthf.cols, CV_16UC1);
  for (int i = 0; i < depthf.rows; ++i) {
    for (int j = 0; j < depthf.cols; ++j) {
      float z = depthf.at<float>(i, j);
      if (std::isnan(z) || std::isinf(z)) {
        depths.at<unsigned short>(i, j) = 0;
      } else {
        float clip_z = 2 * z - 1; // [0,1] -> [-1,1]
        // [-(n+f)/(n-f)] + [2nf/(n-f)] / w_z = clip_z
        GLfloat world_z = 2*z_near_*z_far_/(clip_z*(z_near_-z_far_)+
                                            (z_near_+z_far_));
        float d = world_z * factor;
        d = (d > factor * 5) ? factor * 5 : d;
        depths.at<unsigned short>(i, j) = (unsigned short)(d);
      }
    }
  }
  return depths;
}
}