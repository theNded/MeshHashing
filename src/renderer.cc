//
// Created by wei on 17-3-20.
//

#include <matrix.h>
#include "renderer.h"

////////////////////
/// class Renderer
////////////////////

bool Renderer::is_cuda_init_ = false;

Renderer::Renderer(std::string name, uint width, uint height) {
  width_ = width;
  height_ = height;
  gl_context_.Init(width, height, name);
  is_gl_init_ = true;

  control_ = new gl_utils::Control(gl_context_.window(),
                                   gl_context_.width(),
                                   gl_context_.height());

  m_ = glm::mat4(1.0f);
  m_[1][1] = -1;
  m_[2][2] = -1;

  p_ = gl_context_.projection_mat();
  InitCUDA();
}

Renderer::~Renderer() {}

void Renderer::InitCUDA() {
  if (is_cuda_init_) return;

  /// !!! We assume that the Rendering & Compting are performed
  /// !!! on the same device
  uint gl_device_count = 2;
  int gl_device[2];
  cudaDeviceProp device_prop;
  checkCudaErrors(cudaGLGetDevices(&gl_device_count, gl_device,
                                   gl_device_count,
                                   cudaGLDeviceListAll));
  for (uint i = 0; i < gl_device_count; ++i) {
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, gl_device[i]));
    LOG(INFO) << "Device id: " << gl_device[i]
              << ", name: " << device_prop.name
              << ", with compute capability "
              << device_prop.major << '.' << device_prop.minor;
  }
  cudaSetDevice(gl_device[0]);
  is_cuda_init_ = true;
}

void Renderer::ScreenCapture(unsigned char* data, int width, int height) {
  CHECK(data) << "Invalid image ptr!";

  glReadBuffer(GL_FRONT);
  glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, data);
}

void Renderer::Render(float4x4 cTw) {
  cTw = cTw.getTranspose();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      v_[i][j] = cTw.entries2[i][j];
  if (free_walk_) {
    control_->UpdateCameraPose();
    v_ = control_->view_mat();
  } else {
    v_ = m_ * v_ * glm::inverse(m_);
  }

  glClearColor(1, 1, 1, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  control_->UpdateCameraPose();

  for (auto& object : objects_) {
    object->Render(m_, v_, p_);
  }
  glfwSwapBuffers(gl_context_.window());
  glfwPollEvents();

  if (glfwGetKey(gl_context_.window(), GLFW_KEY_ESCAPE) == GLFW_PRESS ) {
    exit(0);
  }
}

////////////////////
/// class GLObjectBase
////////////////////
void GLObjectBase::CompileShader(std::string vert_glsl_path,
                                 std::string frag_glsl_path,
                                 std::vector<std::string>& uniform_names) {
  gl_utils::LoadShaders(vert_glsl_path, frag_glsl_path, program_);
  for (auto &uniform_name : uniform_names) {
    int uniform = glGetUniformLocation(program_, uniform_name.c_str());
    LOG(INFO) << uniform << " : " << uniform_name.c_str();
    CHECK(uniform >= 0) << "Invalid uniform! ";
    uniforms_.push_back((uint)uniform);
  }
}

////////////////////
/// class Frame
////////////////////
const GLfloat FrameObject::kVertices[8] = {
        -1.0f, -1.0f,     -1.0f,  1.0f,
        1.0f,  1.0f,      1.0f, -1.0f
};
const GLubyte FrameObject::kIndices[6] = {
        0, 1, 2,
        0, 2, 3
};

FrameObject::FrameObject(uint width, uint height) {
  width_ = width;
  height_ = height;

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
               width, height, 0,
               GL_RGBA, GL_FLOAT, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

  /// Bind texture to cuda resources
  checkCudaErrors(cudaGraphicsGLRegisterImage(
          &cuda_resource_, texture_, GL_TEXTURE_2D,
          cudaGraphicsRegisterFlagsNone));
}

FrameObject::~FrameObject() {
  glDeleteTextures(1, &texture_);
  glDeleteBuffers(2, vbo_);
  glDeleteVertexArrays(1, &vao_);
  glDeleteProgram(program_);

  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource_));
  delete[] vbo_;
}

void FrameObject::SetData(float4 *image) {
  cudaArray_t in_array;

  LOG(INFO) << "Transfering from CUDA to OpenGL";
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource_));
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&in_array, cuda_resource_, 0, 0));
  checkCudaErrors(cudaMemcpyToArray(in_array, 0, 0,
                                    image,
                                    sizeof(float4) *
                                    width_ * height_,
                                    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource_, 0));
}

void FrameObject::Render(glm::mat4 m, glm::mat4 v, glm::mat4 p) {
  LOG(INFO) << "OpenGL rendering";
  glUseProgram(program_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glUniform1i(uniforms_[0], 0);
  glBindVertexArray(vao_);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, 0);
}

////////////////////
/// class Mesh
////////////////////
MeshObject::MeshObject(int max_vertex_count, int max_triangle_count,
                       MeshType type) {
  type_ = type;
  max_vertex_count_ = max_vertex_count;
  max_triangle_count_ = max_triangle_count;

  /// 1. Select shader
  std::vector<std::string> uniform_names;
  uniform_names.clear();
  switch (type) {
    case kNormal:
      uniform_names.push_back("mvp");
      uniform_names.push_back("view_mat");
      uniform_names.push_back("model_mat");

      CompileShader("../shader/mesh_vn_vertex.glsl",
                    "../shader/mesh_vn_fragment.glsl",
                    uniform_names);
      break;
    case kColor:
      uniform_names.push_back("mvp");

      CompileShader("../shader/mesh_vc_vertex.glsl",
                    "../shader/mesh_vc_fragment.glsl",
                    uniform_names);
      break;
    default:
      break;
  }

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  /// 2. Generate buffers
  vbo_count_ = 0;
  switch (type) {
    case kNormal:
      vbo_count_ = 3;
      break;
    case kColor:
      vbo_count_ = 3;
      break;
    default:
      vbo_count_ = 0;
      break;
  }

  vbo_ = new GLuint[vbo_count_];
  glGenBuffers(vbo_count_, vbo_);

  /// 3. Bind buffers
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, max_vertex_count * sizeof(float3),
               NULL, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
          &cuda_vertices_, vbo_[0], cudaGraphicsMapFlagsNone));

  switch (type) {
    case kNormal:
      glEnableVertexAttribArray(1);
      glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);
      glBufferData(GL_ARRAY_BUFFER, max_vertex_count * sizeof(float3),
                   NULL, GL_STATIC_DRAW);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
      checkCudaErrors(cudaGraphicsGLRegisterBuffer(
              &cuda_normals_, vbo_[1], cudaGraphicsMapFlagsNone));

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[2]);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, max_triangle_count * sizeof(int3),
                   NULL, GL_STATIC_DRAW);
      checkCudaErrors(cudaGraphicsGLRegisterBuffer(
              &cuda_triangles_, vbo_[2], cudaGraphicsMapFlagsNone));
      break;

    case kColor:
      glEnableVertexAttribArray(1);
      glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);
      glBufferData(GL_ARRAY_BUFFER, max_vertex_count * sizeof(float3),
                   NULL, GL_STATIC_DRAW);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
      checkCudaErrors(cudaGraphicsGLRegisterBuffer(
              &cuda_colors_, vbo_[1], cudaGraphicsMapFlagsNone));

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[2]);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, max_triangle_count * sizeof(int3),
                   NULL, GL_STATIC_DRAW);
      checkCudaErrors(cudaGraphicsGLRegisterBuffer(
              &cuda_triangles_, vbo_[2], cudaGraphicsMapFlagsNone));
      break;

    default:
      break;
  }

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
}

MeshObject::~MeshObject() {
  glDeleteProgram(program_);
  glDeleteBuffers(vbo_count_, vbo_);
  glDeleteVertexArrays(1, &vao_);

  switch (type_) {
    case kNormal:
      checkCudaErrors(cudaGraphicsUnregisterResource(cuda_normals_));
      break;
    case kColor:
      checkCudaErrors(cudaGraphicsUnregisterResource(cuda_colors_));
      break;
    default:
      break;
  }
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vertices_));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_triangles_));

  delete[] vbo_;
}

void MeshObject::SetData(float3 *vertices, size_t vertex_count,
                         float3 *normals,  size_t normal_count,
                         float3 *colors,  size_t color_count,
                         int3 *triangles,  size_t triangle_count) {
  vertex_count_ = vertex_count;
  triangle_count_ = triangle_count;

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
  switch (type_) {
    case kNormal:
      checkCudaErrors(cudaGraphicsMapResources(1, &cuda_normals_));
      checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
              (void **) &map_ptr, &map_size, cuda_normals_));
      checkCudaErrors(cudaMemcpy(map_ptr, normals,
                                 normal_count * sizeof(float3),
                                 cudaMemcpyDeviceToDevice));
      checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_normals_, 0));
      break;
    case kColor:
      checkCudaErrors(cudaGraphicsMapResources(1, &cuda_colors_));
      checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
              (void **) &map_ptr, &map_size, cuda_colors_));
      checkCudaErrors(cudaMemcpy(map_ptr, colors,
                                 color_count * sizeof(float3),
                                 cudaMemcpyDeviceToDevice));
      checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_colors_, 0));
      break;
    default:
      break;
  }

  map_ptr = NULL;
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_triangles_));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
          (void **)&map_ptr, &map_size, cuda_triangles_));
  checkCudaErrors(cudaMemcpy(map_ptr, triangles,
                             triangle_count * sizeof(int3),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_triangles_, 0));
}

void MeshObject::Render(glm::mat4 m, glm::mat4 v, glm::mat4 p) {
  glm::mat4 mvp = p * v * m;
  glUseProgram(program_);

  switch (type_) {
    case kNormal:
      glUniformMatrix4fv(uniforms_[0], 1, GL_FALSE, &mvp[0][0]);
      glUniformMatrix4fv(uniforms_[1], 1, GL_FALSE, &v[0][0]);
      glUniformMatrix4fv(uniforms_[2], 1, GL_FALSE, &m[0][0]);
      break;
    case kColor:
      glUniformMatrix4fv(uniforms_[0], 1, GL_FALSE, &mvp[0][0]);
      break;
    default:
      break;
  }

  glBindVertexArray(vao_);

  // If render mesh only:
  if (ploygon_mode_) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  }
  /// NOTE: Use GL_UNSIGNED_INT instead of GL_INT, otherwise it won't work
  glDrawElements(GL_TRIANGLES, triangle_count_ * 3, GL_UNSIGNED_INT, 0);
}

////////////////////
/// class LineRenderer
////////////////////
LineObject::LineObject(int max_vertex_count) {
  max_vertex_count_ = max_vertex_count;

  std::vector<std::string> uniform_names;
  uniform_names.clear();
  uniform_names.push_back("mvp");

  CompileShader("../shader/line_vertex.glsl",
                "../shader/line_fragment.glsl",
                uniform_names);

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  vbo_ = new GLuint[1];
  glGenBuffers(1, vbo_);

  /// Vertex positions
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, max_vertex_count * sizeof(float3),
               NULL, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
          &cuda_vertices_, vbo_[0], cudaGraphicsMapFlagsNone));

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
}

LineObject::~LineObject(){
  glDeleteProgram(program_);
  glDeleteBuffers(1, vbo_);
  glDeleteVertexArrays(1, &vao_);

  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vertices_));

  delete[] vbo_;
}

void LineObject::SetData(float3 *vertices, size_t vertex_count) {
  vertex_count_ = vertex_count;

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
}

void LineObject::Render(glm::mat4 m, glm::mat4 v, glm::mat4 p) {
  LOG(INFO) << "OpenGL rendering";

  glm::mat4 mvp = p * v * m;
  glUseProgram(program_);
  glUniformMatrix4fv(uniforms_[0], 1, GL_FALSE, &mvp[0][0]);
  glBindVertexArray(vao_);

  glDrawArrays(GL_LINES, 0, vertex_count_);
}

///////////////////
/// class PointObject
///////////////////
PointObject::PointObject(int max_vertex_count) {
  max_vertex_count_ = max_vertex_count;

  std::vector<std::string> uniform_names;
  uniform_names.clear();
  uniform_names.push_back("mvp");

  CompileShader("../shader/point_vertex.glsl",
                "../shader/point_fragment.glsl",
                uniform_names);

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  vbo_ = new GLuint[2];
  glGenBuffers(2, vbo_);

  /// Vertex positions
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, max_vertex_count * sizeof(float3),
               NULL, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ARRAY_BUFFER, max_vertex_count * sizeof(float3),
               NULL, GL_STATIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
          &cuda_vertices_, vbo_[0], cudaGraphicsMapFlagsNone));
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
          &cuda_colors_, vbo_[1], cudaGraphicsMapFlagsNone));

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glDepthFunc(GL_LESS);
}

PointObject::~PointObject() {
  glDeleteProgram(program_);
  glDeleteBuffers(1, vbo_);
  glDeleteVertexArrays(1, &vao_);

  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vertices_));

  delete[] vbo_;
}

void PointObject::SetData(float3 *vertices, size_t vertex_count,
                          float3 *colors, size_t color_count) {
  vertex_count_ = vertex_count;

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
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_colors_));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
          (void **)&map_ptr, &map_size, cuda_colors_));
  checkCudaErrors(cudaMemcpy(map_ptr, colors,
                             color_count * sizeof(float3),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_colors_, 0));
}

void PointObject::Render(glm::mat4 m, glm::mat4 v, glm::mat4 p) {
  LOG(INFO) << "OpenGL rendering";

  glm::mat4 mvp = p * v * m;
  glUseProgram(program_);
  glUniformMatrix4fv(uniforms_[0], 1, GL_FALSE, &mvp[0][0]);
  glBindVertexArray(vao_);

  glDrawArrays(GL_POINTS, 0, vertex_count_);
}