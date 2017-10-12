//
// Created by Neo on 14/08/2017.
//

#include "program.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <GL/glew.h>

namespace gl {
Program::Program(std::string vert_shader_path,
                 std::string frag_shader_path) {
  Build(vert_shader_path, frag_shader_path);
  program_built_ = true;
}

Program::~Program() {
  if (program_built_) {
    glDeleteProgram(program_id_);
  }
}

std::string Program::Load(std::string shader_path) {
  std::string shader_str = "";
  std::ifstream shader_stream(shader_path, std::ios::in);
  if (shader_stream.is_open()) {
    for (std::string line; std::getline(shader_stream, line);) {
      shader_str += line + "\n";
    }
    shader_stream.close();
  } else {
    std::cerr << "Invalid path: " << shader_path << std::endl;
    exit(1);
  }
  return shader_str;
}

GLint Program::Compile(const std::string &shader_str, GLuint &shader_id) {
  GLint result = GL_FALSE;

  GLchar const *shader_cstr[] = {shader_str.c_str()};

  glShaderSource(shader_id, 1, shader_cstr, NULL);
  glCompileShader(shader_id);

  int info_log_length;
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
  glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
  if (info_log_length > 0) {
    std::vector<char> shader_error_msg(info_log_length + 1);
    glGetShaderInfoLog(shader_id, info_log_length, NULL,
                       shader_error_msg.data());
    std::cout << std::string(shader_error_msg.data()) << std::endl;
  }
  return result;
}

GLint Program::Link(GLuint &program_id,
                    GLuint &vert_shader_id,
                    GLuint &frag_shader_id) {
  GLint result = GL_FALSE;
  glAttachShader(program_id, vert_shader_id);
  glAttachShader(program_id, frag_shader_id);
  glLinkProgram(program_id);

  int info_log_length;
  glGetProgramiv(program_id, GL_LINK_STATUS, &result);
  glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);
  if (info_log_length > 0) {
    std::vector<char> program_error_msg(info_log_length + 1);
    glGetProgramInfoLog(program_id, info_log_length, NULL,
                        program_error_msg.data());
    std::cout << std::string(program_error_msg.data()) << std::endl;
  }
  return result;
}

void Program::Build(std::string vert_shader_path,
                    std::string frag_shader_path) {

  // Create the shaders
  program_id_ = glCreateProgram();
  GLuint vert_shader_id = glCreateShader(GL_VERTEX_SHADER);
  GLuint frag_shader_id = glCreateShader(GL_FRAGMENT_SHADER);

  GLint compile_result;
  std::cout << "Compiling vertex shader: " << vert_shader_path << std::endl;
  std::string vert_shader_str = Load(vert_shader_path);
  compile_result = Compile(vert_shader_str, vert_shader_id);
  if (GL_FALSE == compile_result) {
    std::cerr << "Compile error, abort." << std::endl;
    exit(1);
  }

  std::cout << "Compiling fragment shader: " << frag_shader_path << std::endl;
  std::string frag_shader_src = Load(frag_shader_path);
  compile_result = Compile(frag_shader_src, frag_shader_id);
  if (GL_FALSE == compile_result) {
    std::cerr << "Compile error, abort." << std::endl;
    exit(1);
  }

  std::cout << "Linking program ..." << std::endl;
  GLint link_result = Link(program_id_, vert_shader_id, frag_shader_id);
  if (GL_FALSE == link_result) {
    std::cerr << "Link error, abort." << std::endl;
    exit(1);
  }

  glDetachShader(program_id_, vert_shader_id);
  glDetachShader(program_id_, frag_shader_id);

  glDeleteShader(vert_shader_id);
  glDeleteShader(frag_shader_id);

  std::cout << "Success." << std::endl;

  program_built_ = true;
}
}