//
// Created by Neo on 15/08/2017.
//

#include "uniform.h"
#include <iostream>

namespace gl {
Uniform::Uniform(GLuint program, std::string name,
                 UniformType type) {
  GetLocation(program, name);
  set_type(type);
}

void Uniform::GetLocation(GLuint program,
                          std::string name) {
  GLint uniform_id = glGetUniformLocation(program, name.c_str());
  if (uniform_id < 0) {
    std::cerr << "Invalid uniform name!" << std::endl;
    exit(1);
  }
  uniform_id_ = (GLuint)uniform_id;
}

/// Override specially designed for texture
void Uniform::Bind(GLuint id) {
  switch (type_) {
    case kTexture2D:
      glUniform1i(uniform_id_, id);
      break;
    default:
      break;
  }
}
void Uniform::Bind(void *data) {
  switch (type_) {
    case kMatrix4f:
      glUniformMatrix4fv(uniform_id_, 1, GL_FALSE, (float*)data);
      break;
    case kVector3f:
      glUniform3fv(uniform_id_, 1, (float*)data);
    default:
      break;
  }
}


}