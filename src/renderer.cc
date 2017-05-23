//
// Created by wei on 17-3-20.
//

#include <matrix.h>
#include "renderer.h"

////////////////////
/// class RendererBase
////////////////////

bool RendererBase::is_cuda_init_ = false;

RendererBase::RendererBase(std::string name, uint width, uint height) {
  gl_context_ = new gl_utils::Context(name, width, height);
  is_gl_init_ = true;
}

RendererBase::~RendererBase() {
  if (is_gl_init_) delete gl_context_;
}

void RendererBase::InitCUDA() {
  if (is_cuda_init_) return;

  /// !!! We assume that the Rendering & Compting are performed
  /// !!! on the same device
  uint gl_device_count = 2;
  int gl_device[2];
  cudaDeviceProp device_prop;
  checkCudaErrors(cudaGLGetDevices(&gl_device_count, gl_device,
                                   gl_device_count,
                                   cudaGLDeviceListAll));
  for (int i = 0; i < gl_device_count; ++i) {
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, gl_device[i]));
    LOG(INFO) << "Device id: " << gl_device[i]
              << ", name: " << device_prop.name
              << ", with compute capability "
              << device_prop.major << '.' << device_prop.minor;
  }
  cudaSetDevice(gl_device[0]);
  is_cuda_init_ = true;
}

void RendererBase::CompileShader(std::string vert_glsl_path,
                                 std::string frag_glsl_path,
                                 std::vector<std::string>& uniform_names) {
  CHECK(is_gl_init_) << "OpenGL not initialized";

  gl_utils::LoadShaders(vert_glsl_path, frag_glsl_path, program_);
  for (auto &uniform_name : uniform_names) {
    int uniform = glGetUniformLocation(program_, uniform_name.c_str());
    CHECK(uniform >= 0) << "Invalid uniform!";
    uniforms_.push_back((uint)uniform);
  }
}

////////////////////
/// class FrameRenderer
////////////////////
const GLfloat FrameRenderer::kVertices[8] = {
        -1.0f, -1.0f,     -1.0f,  1.0f,
        1.0f,  1.0f,      1.0f, -1.0f
};
const GLubyte FrameRenderer::kIndices[6] = {
        0, 1, 2,
        0, 2, 3
};

FrameRenderer::FrameRenderer(std::string name, uint width, uint height)
        : RendererBase(name, width, height) {
  CHECK(is_gl_init_) << "OpenGL not initialized";

  InitCUDA();

  /// VAO: variable groups
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  /// VBO: variables (separately)
  vbo_ = new GLuint[2];
  glGenBuffers(2, vbo_);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(kIndices), kIndices, GL_STATIC_DRAW);

  /// Texture
  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  ///////////////////////////!!! GL_RGBA32F !!!
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
               gl_context_->width(), gl_context_->height(), 0,
               GL_RGBA, GL_FLOAT, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

  /// Bind texture to cuda resources
  checkCudaErrors(cudaGraphicsGLRegisterImage(
          &cuda_resource_, texture_, GL_TEXTURE_2D,
          cudaGraphicsRegisterFlagsNone));
}

FrameRenderer::~FrameRenderer() {
  glDeleteTextures(1, &texture_);
  glDeleteBuffers(2, vbo_);
  glDeleteVertexArrays(1, &vao_);
  glDeleteProgram(program_);

  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource_));
  delete[] vbo_;
}

void FrameRenderer::Render(float4 *image) {
  cudaArray_t in_array;

  LOG(INFO) << "Transfering from CUDA to OpenGL";
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource_));
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&in_array, cuda_resource_, 0, 0));
  checkCudaErrors(cudaMemcpyToArray(in_array, 0, 0,
                                    image,
                                    sizeof(float4) *
                                    gl_context_->width() * gl_context_->height(),
                                    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource_, 0));

  LOG(INFO) << "OpenGL rendering";
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(program_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glUniform1i(uniforms_[0], 0);
  glBindVertexArray(vao_);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, 0);

  glfwSwapBuffers(gl_context_->window());
  glfwPollEvents();
}

////////////////////
/// class FrameRenderer
////////////////////
MeshRenderer::MeshRenderer(std::string name, uint width, uint height)
        : RendererBase(name, width, height) {
  const int kMaxVertices  = 10000000;
  const int kMaxTriangles = 10000000;

  CHECK(is_gl_init_) << "OpenGL not initialized";
  InitCUDA();

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  vbo_ = new GLuint[2];
  glGenBuffers(2, vbo_);

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, kMaxVertices * sizeof(float3),
               NULL, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, kMaxTriangles * sizeof(int3),
               NULL, GL_STATIC_DRAW);

  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
          &cuda_vertices_, vbo_[0], cudaGraphicsMapFlagsNone));
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
          &cuda_triangles_, vbo_[1], cudaGraphicsMapFlagsNone));

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  control_ = new gl_utils::Control(gl_context_->window(),
                                   gl_context_->width(),
                                   gl_context_->height());
}

MeshRenderer::~MeshRenderer() {
  glDeleteProgram(program_);
  glDeleteBuffers(2, vbo_);
  glDeleteVertexArrays(1, &vao_);

  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vertices_));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_triangles_));

  delete[] vbo_;
  delete[] control_;
}

void MeshRenderer::Render(float3 *vertices, size_t vertex_count,
                          int3 *triangles, size_t triangle_count,
                          float4x4 cTw) {

  LOG(INFO) << "Transfering from CUDA to OpenGL";
  float3 *map_ptr;
  size_t map_size;

  map_ptr = NULL;
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vertices_));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
          (void **)&map_ptr, &map_size, cuda_vertices_));
  checkCudaErrors(cudaMemcpy(map_ptr, vertices,
                             vertex_count * sizeof(float3),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vertices_, 0));

  map_ptr = NULL;
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_triangles_));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
          (void **)&map_ptr, &map_size, cuda_triangles_));
  checkCudaErrors(cudaMemcpy(map_ptr, triangles,
                             triangle_count * sizeof(int3),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_triangles_, 0));

  LOG(INFO) << "OpenGL rendering";
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(program_);

  glm::mat4 transform = glm::mat4(1);
  transform[1][1] = -1;
  transform[2][2] = -1;

  glm::mat4 view_mat;
  cTw = cTw.getTranspose();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      view_mat[i][j] = cTw.entries2[i][j];

  glm::mat4 mvp;
  if (free_walk_) {
    control_->UpdateCameraPose();
    mvp = control_->projection_mat() *
            transform *
            control_->view_mat();
  } else {
    mvp = control_->projection_mat() *
            transform *
            view_mat;// * transform * transform;
  }

  glUniformMatrix4fv(uniforms_[0], 1, GL_FALSE, &mvp[0][0]);
  glBindVertexArray(vao_);
  //glDrawElements(GL_TRIANGLES, triangle_count * 3, GL_UNSIGNED_INT, 0);
  glDrawArrays(GL_POINTS, 0, vertex_count);

  glfwSwapBuffers(gl_context_->window());
  glfwPollEvents();

  if (glfwGetKey(gl_context_->window(), GLFW_KEY_ESCAPE) == GLFW_PRESS ) {
    exit(0);
  }
}