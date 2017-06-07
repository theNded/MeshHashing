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
#include "control.h"

/// this should go after gl_utils
#include "matrix.h"
#include <cuda_gl_interop.h>

class RendererBase {
protected:
  /// Shared
  static bool is_cuda_init_;

  bool is_gl_init_ = false;
  gl_utils::Context gl_context_;

  GLuint vao_;
  GLuint* vbo_;
  GLuint program_;
  std::vector<GLuint> uniforms_;
  GLuint texture_;

public:
  static void InitCUDA();

  RendererBase(gl_utils::Context& context);
  RendererBase(std::string name, uint width, uint height);
  ~RendererBase();
  GLFWwindow* window() {
    return gl_context_.window();
  }
  gl_utils::Context& context() {
    return gl_context_;
  }

  void CompileShader(std::string vert_glsl_path,
                     std::string frag_glsl_path,
                     std::vector<std::string>& uniform_names);
  void ScreenCapture(unsigned char* data, int width, int height);
};

class FrameRenderer : public RendererBase {
private:
  static const GLfloat kVertices[8];
  static const GLubyte kIndices[6];
  cudaGraphicsResource* cuda_resource_;

public:
  FrameRenderer(std::string name, uint width, uint height);
  ~FrameRenderer();
  void Render(float4* image);
};

class MeshRenderer : public RendererBase {
protected:
  bool free_walk_     = false;
  bool line_only_     = false;

  cudaGraphicsResource* cuda_vertices_;
  cudaGraphicsResource* cuda_normals_;
  cudaGraphicsResource* cuda_triangles_;
  gl_utils::Control*    control_;

  int max_vertex_count_;
  int max_triangle_count_;

public:
  bool &free_walk() {
    return free_walk_;
  }
  bool &line_only() {
    return line_only_;
  }

  /// Assume vertex_count == normal_count at current
  MeshRenderer(std::string name, uint width, uint height,
               int max_vertex_count, int triangle_count);
  ~MeshRenderer();
  void Render(float3* vertices, size_t vertex_count,
              float3* normals,  size_t normal_count,
              int3* triangles,  size_t triangle_count,
              float4x4 mvp);
};

class LineRenderer : public RendererBase {
protected:
  bool free_walk_     = false;
  bool line_only_     = false;

  cudaGraphicsResource* cuda_vertices_;
  gl_utils::Control*    control_;

  int max_line_count_;

public:
  bool &free_walk() {
    return free_walk_;
  }
  bool &line_only() {
    return line_only_;
  }

  LineRenderer(gl_utils::Context& context, int max_line_count);
  LineRenderer(std::string name, uint width, uint height, int max_line_count);
  ~LineRenderer();
  void Render(float3* vertices, size_t vertex_count, float4x4 mvp);
};

/// An instance of MeshRenderer for easier use
class MapMeshRenderer : public MeshRenderer {
public:
  MapMeshRenderer(std::string name, uint width, uint height,
                  int max_vertex_count, int triangle_count)
          : MeshRenderer(name, width, height,
                         max_vertex_count, triangle_count) {
    std::vector<std::string> uniform_names;
    uniform_names.clear();
    uniform_names.push_back("mvp");
    uniform_names.push_back("view_mat");
    uniform_names.push_back("model_mat");

    CompileShader("../shader/mesh_vertex.glsl",
                  "../shader/mesh_fragment.glsl",
                  uniform_names);
  }
};

class BBoxRenderer : public LineRenderer {
public:
  BBoxRenderer(gl_utils::Context& context, int max_line_count)
          : LineRenderer(context, max_line_count) {
    std::vector<std::string> uniform_names;
    uniform_names.clear();
    uniform_names.push_back("mvp");

    CompileShader("../shader/line_vertex.glsl",
                  "../shader/line_fragment.glsl",
                  uniform_names);
  }

  BBoxRenderer(std::string name, uint width, uint height, int max_line_count)
          : LineRenderer(name, width, height, max_line_count) {
    std::vector<std::string> uniform_names;
    uniform_names.clear();
    uniform_names.push_back("mvp");

    CompileShader("../shader/line_vertex.glsl",
                  "../shader/line_fragment.glsl",
                  uniform_names);
  }
};
#endif //VH_RENDERER_H
