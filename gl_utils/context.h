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

namespace gl_utils {
class Context {
public:
  Context(std::string window_name, size_t width, size_t height);

  GLFWwindow *window() const;
  size_t width();
  size_t height();

private:
  int Init(std::string window_name);

  GLFWwindow *window_;
  size_t width_;
  size_t height_;
};
}


#endif //RAYTRACING_GLUTILS_CONTEXT_H
