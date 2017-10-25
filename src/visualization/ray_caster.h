//
// Created by wei on 17-3-17.
//

#ifndef VH_RAY_CASTER_H
#define VH_RAY_CASTER_H

#include "core/common.h"
#include "core/params.h"
#include "core/hash_table.h"
#include "core/block_array.h"
#include "geometry/coordinate_utils.h"
#include "sensor/rgbd_sensor.h"

struct RayCasterSample {
  float  entropy;
  float  sdf;
  float  t;
  uint   weight;
};

struct RayCasterData {
  float4 *depth;
  float4 *vertex;
  float4 *normal;
  float4 *color;
  float4 *surface;
};

class RayCaster {
public:
  RayCaster() = default;
  void Alloc(const RayCasterParams &params);
  RayCaster(const RayCasterParams& params);
  void Free();

  void Cast(HashTable& hash_table,
            BlockArray& blocks,
            RayCasterData &ray_caster_data,
            CoordinateConverter& converter,
            const float4x4& c_T_w);

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
  const RayCasterParams& ray_caster_params() const {
    return ray_caster_params_;
  }
  RayCasterData& data() {
    return ray_caster_data_;
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

private:
  bool is_allocated_on_gpu_ = false;
  RayCasterParams  ray_caster_params_;

  cv::Mat depth_image_;
  cv::Mat normal_image_;
  cv::Mat color_image_;
  cv::Mat surface_image_;

  RayCasterData ray_caster_data_;
};

#endif //VH_RAY_CASTER_H
