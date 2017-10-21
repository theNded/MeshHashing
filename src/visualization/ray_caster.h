//
// Created by wei on 17-3-17.
//

#ifndef VH_RAY_CASTER_H
#define VH_RAY_CASTER_H

#include "../core/common.h"

#include "../engine/map.h"
#include "../core/params.h"
#include "../engine/sensor.h"

struct RayCasterSample {
  float  entropy;
  float  sdf;
  float  t;
  uint   weight;
};

struct __ALIGN__(8) RayCasterDataGPU {
  float4 *depth_image;
  float4 *vertex_image;
  float4 *normal_image;
  float4 *color_image;
  float4 *surface_image;
};

class RayCaster {
private:
  RayCasterDataGPU gpu_memory_;
  RayCasterParams  ray_caster_params_;

  cv::Mat          depth_image_;
  cv::Mat          normal_image_;
  cv::Mat          color_image_;
  cv::Mat          surface_image_;

public:
  RayCaster(const RayCasterParams& params);
  ~RayCaster(void);

  void Cast(Map& map, const float4x4& c_T_w);

  const cv::Mat& depth_image() {
    return depth_image_;
  }
  const cv::Mat& normal_image() {
    return normal_image_;
  }
  const cv::Mat& color_image() {
    return color_image_;
  }
  const cv::Mat& surface_image() {
    return surface_image_;
  }
  const RayCasterDataGPU& gpu_memory() {
    return gpu_memory_;
  }
  const RayCasterParams& ray_caster_params() const {
    return ray_caster_params_;
  }

  /// To write images into a video, use this function
  static cv::Mat Mat4fToMat3b(const cv::Mat &mat4f) {
    cv::Mat mat3b = cv::Mat(mat4f.rows, mat4f.cols, CV_8UC3);
    for (int i = 0; i < mat4f.rows; ++i) {
      for (int j = 0; j < mat4f.cols; ++j) {
        cv::Vec4f cf = mat4f.at<cv::Vec4f>(i, j);
        if (std::isinf(cf[0])) {
          mat3b.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
        } else {
          mat3b.at<cv::Vec3b>(i, j) = cv::Vec3b(255 * fabs(cf[0]),
                                                255 * fabs(cf[1]),
                                                255 * fabs(cf[2]));
        }
      }
    }
    return mat3b;
  }
};

#endif //VH_RAY_CASTER_H
