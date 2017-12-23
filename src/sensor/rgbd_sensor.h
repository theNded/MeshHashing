//
// Created by wei on 17-3-20.
//

#ifndef VH_SENSOR_H
#define VH_SENSOR_H

#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include <sophus/se3.hpp>

#include "core/params.h"
#include "sensor/preprocess.h"
#include <glog/logging.h>

struct se3 {
  union {
    struct {
      float t1; float t2; float t3;
      float w1; float w2; float w3;
    };
    float entries[6];
  };
};
/// At first get rid of CUDARGBDAdaptor and Sensor, use it directly
struct SensorData {
  /// sensor data raw
  short*  depth_buffer;
  uchar4* color_buffer;

  /// Reformatted data
  float*	depth_data;
  float*    filtered_depth_data;
  float*    inlier_ratio;
  float4*	color_data;
  float3*   normal_data;

  /// Texture-binded data
  cudaArray*	depth_array;
  cudaArray*	color_array;
  cudaArray*  normal_array;

  cudaTextureObject_t depth_texture;
  cudaTextureObject_t color_texture;
  cudaTextureObject_t normal_texture;

  cudaChannelFormatDesc depth_channel_desc;
  cudaChannelFormatDesc color_channel_desc;
  cudaChannelFormatDesc normal_channel_desc;
};

class Sensor {
public:
  Sensor() = default;
  explicit Sensor(SensorParams &params);
  ~Sensor();
  void BindCUDATexture();

  int Process(cv::Mat &depth, cv::Mat &color);

  void SE3_tangent_to_mat() {
    Eigen::Matrix<float, 6, 1> xi_wTc;
    xi_wTc << xi_wTc_.t1, xi_wTc_.t2, xi_wTc_.t3,
        xi_wTc_.w1, xi_wTc_.w2, xi_wTc_.w3;
    Sophus::SE3f SE3_from_xi = Sophus::SE3f::exp(xi_wTc);
    Eigen::Matrix4f m = SE3_from_xi.matrix();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        wTc_.entries2[i][j] = m.coeff(i, j);
      }
    }
  }

  void SE3_mat_to_tangent() {
    Eigen::Matrix<float, 3, 4> m;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        m.coeffRef(i, j) = wTc_.entries2[i][j];
      }
    }

    Sophus::SE3f SE3_from_mat;
    SE3_from_mat.setRotationMatrix(m.leftCols(3));
    SE3_from_mat.translation() = m.rightCols(1);

    Sophus::SE3f::Tangent xi_ = Sophus::SE3f::log(SE3_from_mat);
    for (int i = 0; i < 6; ++i) {
      xi_wTc_.entries[i] = xi_.coeff(i);
    }
  }

  void set_transform(float4x4 wTc) {
    wTc_ = wTc;
    cTw_ = wTc_.getInverse();
  }
  const se3& w_xi_c() const {
    return xi_wTc_;
  }
  const float4x4& wTc() const {
    return wTc_;
  }
  const float4x4& cTw() const {
    return cTw_;
  }
  uint width() const {
    return params_.width;
  }
  uint height() const {
    return params_.height;
  }
  const SensorData& data() const {
    return data_;
  }
  const SensorParams& sensor_params() const {
    return params_;
  }

private:
  bool is_allocated_on_gpu_ = false;

  /// sensor data
  SensorData	data_;
  SensorParams	params_;

  float4x4      wTc_; // camera -> world
  float4x4      cTw_;

  se3 xi_wTc_;
  se3 dxi_wTc_;
};

#endif //VH_SENSOR_H
