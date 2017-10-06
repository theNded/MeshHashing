//
// Created by wei on 17-3-19.
//

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "../renderer.h"

int main() {
  Renderer renderer("frame", 640, 480);
  std::vector<std::string> uniform_names;
  uniform_names.push_back("texture_sampler");

  FrameObject frame(640, 480);
  frame.CompileShader("../shader/frame_vertex.glsl",
                      "../shader/frame_fragment.glsl",
                      uniform_names);
  renderer.AddObject(&frame);

  ////////// Load data here
  float * cpu_mem;
  float4* cuda_mem;
  cv::Mat im = cv::imread("../unit_test/img.png");

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
    frame.SetData(cuda_mem);

    float4x4 dmy;
    renderer.Render(dmy);
  } while( glfwGetKey(renderer.window(), GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(renderer.window()) == 0);
}
