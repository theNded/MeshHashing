//
// Created by Neo on 15/08/2017.
//

#ifndef OPENGL_SNIPPET_UNIFORM_H
#define OPENGL_SNIPPET_UNIFORM_H

#include <string>
#include <unordered_map>
#include <GL/glew.h>

namespace gl {
enum UniformType {
  kTexture2D,
  kMatrix4f,
  kVector3f,
  kFloat
};

class Uniforms {
public:
  Uniforms() = default;

  void GetLocation(GLuint program,
                   std::string name,
                   UniformType type);

  void Bind(std::string name, GLuint idx);
  void Bind(std::string name, void *data, int n);

  GLuint id(std::string name) {
    return uniform_ids_[name].first;
  }

  UniformType type(std::string name) {
    return uniform_ids_[name].second;
  }

private:
  std::unordered_map<std::string,
      std::pair<GLuint, UniformType> > uniform_ids_;
};
}
#endif //OPENGL_SNIPPET_UNIFORM_H
