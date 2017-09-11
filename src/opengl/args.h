//
// Created by Neo on 14/08/2017.
//

#ifndef OPENGL_SNIPPET_ARGS_H
#define OPENGL_SNIPPET_ARGS_H

/// VAO: arg1, arg2, ..., argn
/// VBO: buf1, buf2, ..., bufn
#include <GL/glew.h>

namespace gl {

struct ArgAttrib {
  GLuint buffer; // ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
  GLint  size;   // sizeof(float),
  GLint  count;  // 1, 2, 3, ...
  GLenum type;   // GL_FLOAT, ...
};

class Args {
public:
  explicit
  Args(int argn);

  ~Args();

  const GLuint vao() const {
    return vao_;
  }
  // i-th buffer, arg attributes, buffer-size
  void InitBuffer(GLuint i,
                  ArgAttrib arg_attrib,
                  size_t max_size);
  void BindBuffer(GLuint i,
                  ArgAttrib arg_attrib,
                  size_t size,
                  void* data);

private:
  int argn_;
  GLuint vao_;
  GLuint *vbos_;
};
}


#endif //OPENGL_SNIPPET_ARGS_H
