//
// Created by wei on 17-3-20.
//

#ifndef VH_RENDERER_H
#define VH_RENDERER_H

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <helper_cuda.h>

#include "shader.h"
#include "context.h"

/// this should go after gl_utils
#include <cuda_gl_interop.h>

class RendererBase {
protected:
  /// Shared
  static bool is_gl_init_;
  static bool is_cuda_init_;
  static gl_utils::Context *gl_context_;

  GLuint vao_;
  GLuint* vbo_;
  GLuint program_;
  std::vector<GLuint> uniforms_;
  GLuint texture_;

public:
  static void InitGLWindow(std::string name, uint width, uint height);
  static void DestroyGLWindow();
  static void InitCUDA();
  static GLFWwindow* window() {
    return gl_context_->window();
  }

  void CompileShader(std::string vert_glsl_path,
                     std::string frag_glsl_path,
                     std::vector<std::string>& uniform_names);
};

class FrameRenderer : public RendererBase {
private:
  static const GLfloat kVertices[8];
  static const GLubyte kIndices[6];
  cudaGraphicsResource* cuda_resource_;

public:
  FrameRenderer();
  ~FrameRenderer();
  void Render(float4* image);
};

class MeshRenderer : public RendererBase {
private:
  cudaGraphicsResource* cuda_vertices_;
  cudaGraphicsResource* cuda_triangles_;

public:
  MeshRenderer();
  ~MeshRenderer();
  void Render(float3* vertices, size_t vertex_count,
              int3* triangles,  size_t triangle_count,
              float* mvp);
};
#endif //VH_RENDERER_H
