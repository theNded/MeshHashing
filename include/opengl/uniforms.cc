//
// Created by Neo on 15/08/2017.
//

#include "uniforms.h"
#include <iostream>

namespace gl {

void Uniforms::GetLocation(GLuint program, std::string name, UniformType type) {
  GLint uniform_id = glGetUniformLocation(program, name.c_str());
  if (uniform_id < 0) {
    std::cerr << "Invalid uniform name!" << std::endl;
    exit(1);
  }
  uniform_ids_[name] = std::make_pair(uniform_id, type);
  std::cout << name << " " << uniform_id << std::endl;
  std::cout << name << " " << id(name) << std::endl;
}

/// Override specially designed for texture
void Uniforms::Bind(std::string name, GLuint idx) {
  switch (type(name)) {
    case kTexture2D:
    std::cout << name << " " << id(name) << std::endl;
      glUniform1i(id(name), idx);
      break;
    default:
      break;
  }
}

void Uniforms::Bind(std::string name, void *data) {
  switch (type(name)) {
    case kMatrix4f:
      glUniformMatrix4fv(id(name), 1, GL_FALSE, (float*)data);
      break;
    case kVector3f:
      glUniform3fv(id(name), 1, (float*)data);
    default:
      break;
  }
}


}