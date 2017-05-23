//
// Created by wei on 17-3-20.
//

#include "renderer.h"

const GLfloat Renderer::kVertices[8] = {
        -1.0f, -1.0f,     -1.0f,  1.0f,
        1.0f,  1.0f,      1.0f, -1.0f
};

const GLubyte Renderer::kIndices[6] = {
        0, 1, 2,
        0, 2, 3
};

Renderer::Renderer() {
  is_gl_initialized_ = false;
  is_cuda_initialized_ = false;
}

Renderer::~Renderer() {
  if (is_cuda_initialized_) {
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource_));
  }

  if (is_gl_initialized_) {
    glDeleteTextures(1, &texture_);
    glDeleteBuffers(2, vbo_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteProgram(program_);

    delete gl_context_;
  }
}

void Renderer::InitGL(std::string window_name,
                      unsigned int width,
                      unsigned int height,
                      std::string vert_glsl_path,
                      std::string frag_glsl_path,
                      std::string texture_sampler_name) {
  if (is_gl_initialized_) return;
  gl_context_ = new gl_utils::Context(window_name, width, height);
  gl_utils::LoadShaders(vert_glsl_path, frag_glsl_path, program_);
  sampler_ = glGetUniformLocation(program_, texture_sampler_name.c_str());

  /// Vertex Array: an array of input params
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  glGenBuffers(2, vbo_);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(kIndices), kIndices, GL_STATIC_DRAW);

  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    ///////////////////////////!!! GL_RGBA32F !!!
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, gl_context_->width(), gl_context_->height(), 0,
                 GL_RGBA, GL_FLOAT, NULL);
  }
  glBindTexture(GL_TEXTURE_2D, 0);

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  is_gl_initialized_ = true;
}

/// This should be done later than initialization
void Renderer::InitCUDA() {
  if (is_cuda_initialized_) return;
  if (! is_gl_initialized_) {
    LOG(ERROR) << "Please InitGL before InitCUDA!";
  }

  int device_id = gpuGetMaxGflopsDeviceId();
  cudaDeviceProp device_prop;
  uint gl_device_count = 2;
  int gl_device[2];
  checkCudaErrors(cudaGLGetDevices(&gl_device_count, gl_device,
                                   gl_device_count,
                                   cudaGLDeviceListAll));
  for (int i = 0; i < gl_device_count; ++i) {
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, gl_device[i]));
    LOG(INFO) << "Device id: " << gl_device[i]
              << ", name: " << device_prop.name
              << ", with compute capability" << device_prop.major << '.' << device_prop.minor;
  }

  checkCudaErrors(cudaSetDevice(0));
  /// Bind GL texture with CUDA resources
  checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_resource_, texture_, GL_TEXTURE_2D,
                                              cudaGraphicsRegisterFlagsNone));
  is_cuda_initialized_ = true;
}

void Renderer::Render(float4 *cuda_mem, bool on_gl) {
  if (on_gl) {
    cudaArray_t in_array;

    /// Load CUDA memory into CUDA resouce
    LOG(INFO) << "Transfering from CUDA to OpenGL";
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource_));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&in_array, cuda_resource_, 0, 0));
    checkCudaErrors(cudaMemcpyToArray(in_array, 0, 0, cuda_mem,
                                      sizeof(float4) * gl_context_->width() * gl_context_->height(),
                                      cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource_, 0));

    LOG(INFO) << "OpenGL rendering";
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFinish();

    glUseProgram(program_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glUniform1i(sampler_, 0);
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, 0);

    glfwSwapBuffers(gl_context_->window());
    glfwPollEvents();
  }
}
