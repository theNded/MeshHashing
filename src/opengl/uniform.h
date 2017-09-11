//
// Created by Neo on 15/08/2017.
//

#ifndef OPENGL_SNIPPET_UNIFORM_H
#define OPENGL_SNIPPET_UNIFORM_H

#include <string>
#include <GL/glew.h>

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
}


#endif //OPENGL_SNIPPET_UNIFORM_H
