//
// Created by Neo on 16/7/17.
// Compile and link vertex and fragment shaders
//

#ifndef GLUTILS_SHADER_H
#define GLUTILS_SHADER_H

#include <string>

#include <GL/glew.h>

namespace gl_utils {
GLint LoadShaders(std::string vertex_shader_path,
                  std::string fragment_shader_path,
                  GLuint     &program_id);
}
#endif //RAYTRACING_SHADER_H
