//
// Created by Neo on 14/08/2017.
//

#include "args.h"

#include <string>
#include <vector>
#include <cassert>
#include <GL/glew.h>

namespace gl {
Args::Args(int argn) {
  argn_ = argn;

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  vbos_ = new GLuint[argn_];
  glGenBuffers(argn_, vbos_);
}

Args::~Args() {
  glDeleteBuffers(argn_, vbos_);
  glDeleteVertexArrays(1, &vao_);
}

void Args::InitBuffer(GLuint i,
                      ArgAttrib arg_attrib,
                      size_t max_size) {
  assert(i < argn_);
  /// Now we have only ELEMENT_BUFFER and ELEMENT_ARRAY_BUFFER
  if (arg_attrib.buffer != GL_ELEMENT_ARRAY_BUFFER) {
    glEnableVertexAttribArray(i);
  }

  glBindBuffer(arg_attrib.buffer, vbos_[i]);
  glBufferData(arg_attrib.buffer,
               arg_attrib.size * arg_attrib.count * max_size,
               NULL,
               GL_STATIC_DRAW);

  if (arg_attrib.buffer != GL_ELEMENT_ARRAY_BUFFER) {
    glVertexAttribPointer(i, arg_attrib.count, arg_attrib.type,
                          GL_FALSE, 0, NULL);
  }
}

void Args::BindBuffer(GLuint i,
                      ArgAttrib arg_attrib,
                      size_t size,
                      void *data) {
  assert(i < argn_);
  /// Now we have only ELEMENT_BUFFER and ELEMENT_ARRAY_BUFFER
  if (arg_attrib.buffer != GL_ELEMENT_ARRAY_BUFFER) {
    glEnableVertexAttribArray(i);
  }

  glBindBuffer(arg_attrib.buffer, vbos_[i]);
  glBufferData(arg_attrib.buffer,
               arg_attrib.size * arg_attrib.count * size,
               data,
               GL_STATIC_DRAW);

  if (arg_attrib.buffer != GL_ELEMENT_ARRAY_BUFFER) {
    glVertexAttribPointer(i, arg_attrib.count, arg_attrib.type,
                          GL_FALSE, 0, NULL);
  }
}
}