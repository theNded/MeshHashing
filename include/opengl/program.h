//
// Created by Neo on 14/08/2017.
//

#ifndef OPENGL_SNIPPET_SHADER_H
#define OPENGL_SNIPPET_SHADER_H

#include <string>
#include <GL/glew.h>
namespace gl {

const int kShaderTypes = 2;
enum ShaderType {
  kVertexShader = 0,
  kFragmentShader = 1,
};

class Program {
public:
  /// load & replace => Build (compile & link)
  Program() = default;
  ~Program();

  void Load(std::string shader_path, ShaderType type);

  /// Beta function:
  /// Specifically replace the 1st:
  /// #define MACRO 0 => #define MACRO value
  void ReplaceMacro(std::string name, std::string value,
                    ShaderType type);

  void Build();

  const GLuint id() const {
    return program_id_;
  }

private:
  std::string shader_path_[kShaderTypes];
  std::string shader_str_[kShaderTypes];

  GLint Compile(const std::string& shader_str, GLuint &shader_id);
  GLint Link(GLuint &program_id,
             GLuint &vert_shader_id,
             GLuint &frag_shader_id);

  bool program_built_ = false;
  GLuint program_id_;
};
}


#endif //OPENGL_SNIPPET_SHADER_H
