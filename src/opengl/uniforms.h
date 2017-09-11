//
// Created by Neo on 15/08/2017.
//

#ifndef OPENGL_SNIPPET_UNIFORM_H
#define OPENGL_SNIPPET_UNIFORM_H

#include <string>
#include <unordered_map>
#include <GL/glew.h>

#include "uniform.h"

namespace gl {

class Uniforms {
public:
  Uniforms() = default;

  void GetLocation(GLuint program, std::string name, UniformType type);

  void Bind(std::string name, GLuint idx);
  void Bind(std::string name, void *data);

  GLuint id(std::string name) {
    return uniform_ids_[name].first;
  }
  UniformType type(std::string name) {
    return uniform_ids_[name].second;
  }

private:
  UniformType type_;
  std::unordered_map<std::string, std::pair<GLuint, UniformType> > uniform_ids_;
};
}


#endif //OPENGL_SNIPPET_UNIFORM_H
