//
// Created by wei on 17-9-11.
//

#ifndef VOXEL_HASHING_UNIFORM_H
#define VOXEL_HASHING_UNIFORM_H

#include <string>
#include <GL/glew.h>
#include <iostream>

namespace gl {
enum UniformType {
  kTexture2D,
  kMatrix4f,
  kVector3f
};

class Uniform {
public:
  Uniform() = default;
  explicit
  Uniform(GLuint program,
          std::string name,
          UniformType type);

  void GetLocation(GLuint program,
                   std::string name);
  void set_type(UniformType type) {
    type_ = type;
  }

  void Bind(GLuint id);
  void Bind(void *data);

  const GLuint id() const {
    return uniform_id_;
  }

private:
  UniformType type_;
  GLuint uniform_id_;
};

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



#endif //VOXEL_HASHING_UNIFORM_H
