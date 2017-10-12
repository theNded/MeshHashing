//
// Created by Neo on 14/08/2017.
//

#ifndef OPENGL_SNIPPET_SHADER_H
#define OPENGL_SNIPPET_SHADER_H

#include <string>
#include <GL/glew.h>
namespace gl {
class Program {
public:
  Program() = default;
  Program(std::string vert_shader_path,
          std::string frag_shader_path);
  ~Program();

  void Build(std::string vert_shader_path,
             std::string frag_shader_path);
  const GLuint id() const {
    return program_id_;
  }

private:
  std::string Load(std::string shader_path);
  GLint Compile(const std::string& shader_str, GLuint &shader_id);
  GLint Link(GLuint &program_id,
             GLuint &vert_shader_id,
             GLuint &frag_shader_id);

  bool program_built_ = false;
  GLuint program_id_;
};
}


#endif //OPENGL_SNIPPET_SHADER_H
