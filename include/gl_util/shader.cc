//
// Created by Neo on 16/7/17.
//


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <GL/glew.h>

#include "shader.h"

namespace gl_utils {
std::string ReadShader(std::string file_path) {
  std::string shader_src = "";
  std::ifstream shader_stream(file_path, std::ios::in);
  if (shader_stream.is_open()) {
    for (std::string line; std::getline(shader_stream, line);) {
      shader_src += line + "\n";
    }
    shader_stream.close();
  } else {
    LOG(ERROR) << "Invalid path: " << file_path;
  }
  return shader_src;
}

GLint CompileShader(const std::string &shader_src,
                    GLuint &shader_id) {
  GLint result = GL_FALSE;
  int info_log_length;

  GLchar const *shader_src_glchar[] = {shader_src.c_str()};
  glShaderSource(shader_id, 1, shader_src_glchar, NULL);
  glCompileShader(shader_id);

  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
  glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
  if (info_log_length > 0) {
    std::vector<char> shader_error_msg(info_log_length + 1);
    glGetShaderInfoLog(shader_id, info_log_length, NULL,
                       shader_error_msg.data());
    LOG(ERROR) << std::string(shader_error_msg.data());
  }
  return result;
}

GLint LoadShaders(std::string vertex_shader_path,
                  std::string fragment_shader_path,
                  GLuint &program_id) {

  // Create the shaders
  program_id = glCreateProgram();
  GLuint vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
  GLuint fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);

  /// Compile shaders
  GLint compile_result = GL_FALSE;
  LOG(INFO) << "Compiling vertex shader: " << vertex_shader_path;
  std::string vertex_shader_src = ReadShader(vertex_shader_path);
  compile_result = CompileShader(vertex_shader_src, vertex_shader_id);
  if (GL_FALSE == compile_result) {
    LOG(ERROR) << "Error compiling vertex shader !";
    return -1;
  }

  LOG(INFO) << "Compiling fragment shader: " << fragment_shader_path;
  std::string fragment_shader_src = ReadShader(fragment_shader_path);
  compile_result = CompileShader(fragment_shader_src, fragment_shader_id);
  if (GL_FALSE == compile_result) {
    LOG(ERROR) << "Error compiling fragment shader !";
    return -1;
  }

  /// Link shaders
  GLint link_result = GL_FALSE;
  LOG(INFO) << "Linking program ...";
  int info_log_length;
  glAttachShader(program_id, vertex_shader_id);
  glAttachShader(program_id, fragment_shader_id);
  glLinkProgram(program_id);

  glGetProgramiv(program_id, GL_LINK_STATUS, &link_result);
  glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);
  if (info_log_length > 0) {
    std::vector<char> program_error_msg(info_log_length + 1);
    glGetProgramInfoLog(program_id, info_log_length, NULL,
                        program_error_msg.data());
    LOG(ERROR) << std::string(program_error_msg.data());
    return -1;
  }

  glDetachShader(program_id, vertex_shader_id);
  glDetachShader(program_id, fragment_shader_id);

  glDeleteShader(vertex_shader_id);
  glDeleteShader(fragment_shader_id);

  return 0;
}
}