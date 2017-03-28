//
// Created by wei on 17-3-20.
//

#ifndef MRF_VH_RENDERER_H
#define MRF_VH_RENDERER_H

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <helper_cuda.h>

#include "shader.h"
#include "context.h"

/// this should go after gl_utils
#include <cuda_gl_interop.h>

struct Renderer {
private:
  static const GLfloat kVertices[8];
  static const GLubyte kIndices[6];

  gl_utils::Context *gl_context_;

  GLuint vao_;
  GLuint vbo_[2];
  GLuint program_;
  GLuint sampler_;

  /// -----------------------------------------------------
  /// OpenGL part of texture handle
  GLuint texture_;
  /// CUDA part of texture handle
  /// cudaGraphicsResource is a struct
  /// an cudaArray_t inside stores texture data
  /// use cudaGraphicsSubRecourseGetMappedArray to access the data
  cudaGraphicsResource* cuda_resource_;
  /// -----------------------------------------------------

  bool is_gl_initialized_;
  bool is_cuda_initialized_;

public:
  Renderer();
  ~Renderer();

  GLFWwindow* window() {
    return gl_context_->window();
  }

  void InitGL(std::string window_name,
              unsigned int width,
              unsigned int height,
              std::string vert_glsl_path,
              std::string frag_glsl_path,
              std::string texture_sampler_name);
  void InitCUDA();

  /// READ
  void Render(float4 *cuda_mem, bool on_gl = false);
};


#endif //MRF_VH_RENDERER_H
