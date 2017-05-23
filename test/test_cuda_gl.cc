//
// Created by wei on 17-3-19.
//

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "renderer.h"

int main() {
  RendererBase::InitGLWindow("Display", 640, 480);
  RendererBase::InitCUDA();
  FrameRenderer renderer;
  renderer.CompileShader("../shader/vertex.glsl",
                         "../shader/fragment.glsl",
                         "texture_sampler");

  ////////// Load data here
  float * cpu_mem;
  float4* cuda_mem;
  cv::Mat im = cv::imread("../test/img.png");

  cv::resize(im, im, cv::Size(640, 480));
  cpu_mem = new float[4 * sizeof(float) * 640 * 480];
  for (int i = 0; i < im.rows; ++i) {
    for (int j = 0; j < im.cols; ++j) {
      int data_idx = i * im.cols + j;
      cv::Vec3b bgr = im.at<cv::Vec3b>(i, j);
      cpu_mem[4 * data_idx + 0] = bgr[2] / 255.0f;
      cpu_mem[4 * data_idx + 1] = bgr[1] / 255.0f;
      cpu_mem[4 * data_idx + 2] = bgr[0] / 255.0f;
      cpu_mem[4 * data_idx + 3] = 1;
    }
  }

  /// Alloc and memcpy CUDA data here
  checkCudaErrors(cudaMalloc(&cuda_mem, sizeof(float4) * 640 * 480));
  checkCudaErrors(cudaMemcpy(cuda_mem, cpu_mem, sizeof(float4) * 640 * 480,
                             cudaMemcpyHostToDevice));

  LOG(INFO) << "Begin Loop";
  do {
    renderer.Render(cuda_mem);
  } while( glfwGetKey(RendererBase::window(), GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(RendererBase::window()) == 0);
}
