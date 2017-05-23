//
// Created by wei on 17-3-20.
//

#include "renderer.h"

////////////////////
/// class RendererBase
////////////////////
bool RendererBase::is_gl_init_ = false;
bool RendererBase::is_cuda_init_ = false;
gl_utils::Context* RendererBase::gl_context_ = nullptr;

void RendererBase::InitGLWindow(std::string name, uint width, uint height) {
  if (is_gl_init_) return;
  gl_context_ = new gl_utils::Context(name, width, height);
  is_gl_init_ = true;
}

void RendererBase::DestroyGLWindow() {
  if (is_gl_init_) delete gl_context_;
}

void RendererBase::InitCUDA() {
  if (is_cuda_init_) return;
  if (! is_gl_init_) {
    LOG(ERROR) << "Please initialize OpenGL before initialize CUDA";
    return;
  }

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
                                 std::string tex_sampler_name) {
  CHECK(is_gl_init_) << "OpenGL not initialized";

  gl_utils::LoadShaders(vert_glsl_path, frag_glsl_path, program_);
  int sampler = glGetUniformLocation(program_, tex_sampler_name.c_str());
  CHECK(sampler >= 0) << "Invalid sampler!";
  sampler_ = (uint)sampler;
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

FrameRenderer::FrameRenderer() {
  CHECK(is_gl_init_) << "OpenGL not initialized";
  CHECK(is_cuda_init_) << "CUDA not initialized";

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
  checkCudaErrors(
          cudaGraphicsGLRegisterImage(&cuda_resource_, texture_,
                                      GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsNone));
}

FrameRenderer::~FrameRenderer() {
  glDeleteTextures(1, &texture_);
  glDeleteBuffers(2, vbo_);
  glDeleteVertexArrays(1, &vao_);
  glDeleteProgram(program_);

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
  glUniform1i(sampler_, 0);
  glBindVertexArray(vao_);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, 0);

  glfwSwapBuffers(gl_context_->window());
  glfwPollEvents();
}