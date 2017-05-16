//
// Created by wei on 17-4-30.
//

#ifndef VH_CONFIG_READER
#define VH_CONFIG_READER

#include <string>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "matrix.h"
#include "params.h"

void LoadSDFParams(std::string path, SDFParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.voxel_size                = (float)fs["voxel_size"];
  params.sdf_upper_bound           = (float)fs["sdf_upper_bound"];
  params.truncation_distance_scale = (float)fs["truncation_distance_scale"];
  params.truncation_distance       = (float)fs["truncation_distance"];
  params.weight_sample             = (int)fs["weight_sample"];
  params.weight_upper_bound        = (int)fs["weight_upper_bound"];
}

void LoadHashParams(std::string path, HashParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.bucket_count     = (int)fs["bucket_count"];
  params.bucket_size      = (int)fs["bucket_size"];
  params.entry_count      = (int)fs["entry_count"];
  params.linked_list_size = (int)fs["linked_list_size"];
  params.value_capacity   = (int)fs["value_capacity"];
}

void LoadRayCasterParams(std::string path, RayCasterParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.width                = (int)  fs["width"];
  params.height               = (int)  fs["height"];
  params.min_raycast_depth    = (float)fs["min_raycast_depth"];
  params.max_raycast_depth    = (float)fs["max_raycast_depth"];
  params.raycast_step         = (float)fs["raycast_step"];
  params.sample_sdf_threshold = (float)fs["sample_sdf_threshold"];
  params.sdf_threshold        = (float)fs["sdf_threshold"];
}

void LoadSensorParams(std::string path, SensorParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.fx = (float)fs["fx"];
  params.fy = (float)fs["fy"];
  params.cx = (float)fs["cx"];
  params.cy = (float)fs["cy"];
  params.min_depth_range = (float)fs["min_depth_range"];
  params.max_depth_range = (float)fs["max_depth_range"];
  params.range_factor    = (float)fs["range_factor"];
  params.width  = (int)fs["width"];
  params.height = (int)fs["height"];
}

/// 1-1-1 correspondences
void LoadICL(std::string dataset_path,
             std::vector<std::string> &depth_image_list,
             std::vector<std::string> &color_image_list,
             std::vector<float4x4>& wTcs) {
  std::ifstream img_stream(dataset_path + "associations.txt");
  std::string time_stamp, depth_image_name, color_image_name;
  while (img_stream >> time_stamp >> depth_image_name
                    >> time_stamp >> color_image_name) {
    depth_image_list.push_back(dataset_path + "/" + depth_image_name);
    color_image_list.push_back(dataset_path + "/" + color_image_name);
  }

  std::ifstream traj_stream(dataset_path + "traj0.gt.freiburg");
  std::string ts_img, img_path, ts_gt;
  float tx, ty, tz, qx, qy, qz, qw;
  while (traj_stream >> ts_img
                     >> tx >> ty >> tz
                     >> qx >> qy >> qz >> qw) {
    float4x4 wTc;
    wTc.setIdentity();

    wTc.m11 = 1 - 2 * qy * qy - 2 * qz * qz;
    wTc.m12 = 2 * qx * qy - 2 * qz * qw;
    wTc.m13 = 2 * qx * qz + 2 * qy * qw;
    wTc.m14 = tx;
    wTc.m21 = 2 * qx * qy + 2 * qz * qw;
    wTc.m22 = 1 - 2 * qx * qx - 2 * qz * qz;
    wTc.m23 = 2 * qy * qz - 2 * qx * qw;
    wTc.m24 = ty;
    wTc.m31 = 2 * qx * qz - 2 * qy * qw;
    wTc.m32 = 2 * qy * qz + 2 * qx * qw;
    wTc.m33 = 1 - 2 * qx * qx - 2 * qy * qy;
    wTc.m34 = tz;
    wTc.m44 = 1;
    wTcs.push_back(wTc);
  }
}

void LoadSUN3D(std::string dataset_path,
               std::vector<std::string> &depth_img_list,
               std::vector<std::string> &color_img_list,
               std::vector<float4x4> &wTcs) {
  std::ifstream color_stream(dataset_path + "color.txt");
  LOG(INFO) << dataset_path + "color.txt";
  std::string img_name;
  while (color_stream >> img_name) {
    color_img_list.push_back(dataset_path + "color/" + img_name);
  }

  std::ifstream depth_stream(dataset_path + "depth.txt");
  while (depth_stream >> img_name) {
    depth_img_list.push_back(dataset_path + "depth/" + img_name);
  }

  std::ifstream traj_stream(dataset_path + "trajectory.log");
  int dummy;
  float4x4 wTc;
  while (traj_stream >> dummy >> dummy >> dummy
          >> wTc.m11 >> wTc.m12 >> wTc.m13 >> wTc.m14
          >> wTc.m21 >> wTc.m22 >> wTc.m23 >> wTc.m24
          >> wTc.m31 >> wTc.m32 >> wTc.m33 >> wTc.m34
          >> wTc.m41 >> wTc.m42 >> wTc.m43 >> wTc.m44) {
    wTcs.push_back(wTc);
  }
}

/// no 1-1-1 correspondences
void LoadTUM(std::string dataset_path,
             std::vector<std::string> &depth_image_list,
             std::vector<std::string> &color_image_list,
             std::vector<float4x4>& wTcs) {
  std::ifstream img_stream(dataset_path + "depth_rgb_associations.txt");
  std::unordered_map<std::string, std::string> depth_color_correspondence;
  std::string depth_image_name, color_image_name, ts;
  while (img_stream >> ts >> depth_image_name >> ts >> color_image_name) {
    depth_color_correspondence.emplace(depth_image_name, color_image_name);
  }

  std::ifstream traj_stream(dataset_path + "depth_gt_associations.txt");
  float tx, ty, tz, qx, qy, qz, qw;
  while (traj_stream >> ts >> depth_image_name
                     >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
    if (depth_color_correspondence.find(depth_image_name)
        != depth_color_correspondence.end()) {
      float4x4 wTc;
      wTc.setIdentity();

      wTc.m11 = 1 - 2 * qy * qy - 2 * qz * qz;
      wTc.m12 = 2 * qx * qy - 2 * qz * qw;
      wTc.m13 = 2 * qx * qz + 2 * qy * qw;
      wTc.m14 = tx;
      wTc.m21 = 2 * qx * qy + 2 * qz * qw;
      wTc.m22 = 1 - 2 * qx * qx - 2 * qz * qz;
      wTc.m23 = 2 * qy * qz - 2 * qx * qw;
      wTc.m24 = ty;
      wTc.m31 = 2 * qx * qz - 2 * qy * qw;
      wTc.m32 = 2 * qy * qz + 2 * qx * qw;
      wTc.m33 = 1 - 2 * qx * qx - 2 * qy * qy;
      wTc.m34 = tz;
      wTc.m44 = 1;

      depth_image_list.push_back(dataset_path + "/" + depth_image_name);
      color_image_list.push_back(dataset_path + "/"
                                 + depth_color_correspondence[depth_image_name]);
      wTcs.push_back(wTc);
      LOG(INFO) << depth_image_name << " "
                << depth_color_correspondence[depth_image_name];
    }
  }
}

#endif //VH_CONFIG_READER
