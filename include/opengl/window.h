//
// Created by Neo on 14/08/2017.
//

#ifndef OPENGL_SNIPPET_WINDOW_H
#define OPENGL_SNIPPET_WINDOW_H

#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

// set window size
// get screen capture
namespace gl {
class Window {
public:
  Window(std::string window_name, int width, int height);

  /// GLFW operations
  void swap_buffer() {
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
  int should_close() {
    return glfwWindowShouldClose(window_);
  }
  int get_key(int key) {
    return glfwGetKey(window_, key);
  }

  /// Properties
  const int width() const {
    return width_;
  }
  const int height() const {
    return height_;
  }

  /// Screenshot utilities
  cv::Mat CaptureRGB();
  cv::Mat CaptureRGBA();
  cv::Mat CaptureDepth();

private:
  GLFWwindow *window_;
  int width_;
  int height_;

  int img_width_;
  int img_height_;
  cv::Mat rgb_;
  cv::Mat rgba_;
  cv::Mat depth_;
};
}

#endif //OPENGL_SNIPPET_WINDOW_H
