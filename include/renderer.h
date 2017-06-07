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

class GLObjectBase;

class Renderer {
protected:
  /// Shared
  static bool is_cuda_init_;
  bool is_gl_init_ = false;
  static void InitCUDA();

  gl_utils::Context  gl_context_;
  gl_utils::Control *control_;
  uint               width_;
  uint               height_;

  std::vector<GLObjectBase*> objects_;

  bool free_walk_     = false;

  glm::mat4 m_;
  glm::mat4 v_;
  glm::mat4 p_;

public:
  Renderer(std::string name, uint width, uint height);
  ~Renderer();

  void Render(float4x4 cTw);
  void ScreenCapture(unsigned char* data, int width, int height);

  void AddObject(GLObjectBase* object) {
    objects_.push_back(object);
  }

  gl_utils::Context& context() {
    return gl_context_;
  }
  GLFWwindow* window() {
    return gl_context_.window();
  }
  const uint& width() const {
    return width_;
  }
  const uint& height() const {
    return height_;
  }
  bool &free_walk() {
    return free_walk_;
  }
};

class GLObjectBase {
protected:
  GLuint  vao_;
  GLuint* vbo_;
  GLuint  program_;
  GLuint  texture_;

  std::vector<GLuint> uniforms_;

public:
  GLObjectBase(){};
  ~GLObjectBase(){};

  virtual void Render(glm::mat4 m, glm::mat4 v, glm::mat4 p) = 0;
  void CompileShader(std::string vert_glsl_path,
                     std::string frag_glsl_path,
                     std::vector<std::string>& uniform_names);
};

class FrameObject : public GLObjectBase {
protected:
  static const GLfloat kVertices[8];
  static const GLubyte kIndices[6];

  cudaGraphicsResource* cuda_resource_;

  uint width_;
  uint height_;
public:
  FrameObject(uint width, uint height);
  ~FrameObject();

  void Render(glm::mat4 m, glm::mat4 v, glm::mat4 p);
  void SetData(float4* image);
};

enum MeshType {
  kNormal,
  kColor,
  kNormalColor
};

class MeshObject : public GLObjectBase {
protected:
  bool line_only_     = false;

  cudaGraphicsResource* cuda_vertices_;
  cudaGraphicsResource* cuda_normals_;
  cudaGraphicsResource* cuda_colors_;
  cudaGraphicsResource* cuda_triangles_;

  int max_vertex_count_;
  int max_triangle_count_;

  int vertex_count_;
  int triangle_count_;

  int vbo_count_;
  MeshType type_;

public:
  bool &line_only() {
    return line_only_;
  }

  MeshObject(int max_vertex_count, int max_triangle_count,
             MeshType type = kNormal);
  ~MeshObject();
  void Render(glm::mat4 m, glm::mat4 v, glm::mat4 p);
  void SetData(float3* vertices, size_t vertex_count,
               float3* normals,  size_t normal_count,
               int3* triangles,  size_t triangle_count);
};

class LineObject : public GLObjectBase {
protected:
  cudaGraphicsResource* cuda_vertices_;

  int max_vertex_count_;
  int vertex_count_;

public:
  LineObject(int max_vertex_count);
  ~LineObject();
  void Render(glm::mat4 m, glm::mat4 v, glm::mat4 p);
  void SetData(float3* vertices, size_t vertex_count);
};

#endif //VH_RENDERER_H
