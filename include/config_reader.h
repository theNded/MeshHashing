//
// Created by wei on 17-4-30.
//

#ifndef VH_CONFIG_READER
#define VH_CONFIG_READER

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
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
  params.width  = (int)fs["width"];
  params.height = (int)fs["height"];
}

void LoadICLImageList(std::string dataset_path,
                      std::vector<std::string> &depth_image_list,
                      std::vector<std::string> &color_image_list) {
  std::ifstream list_stream(dataset_path + "associations.txt");
  std::string time_stamp, depth_image_name, color_image_name;
  while (list_stream >> time_stamp >> depth_image_name
                     >> time_stamp >> color_image_name) {
    depth_image_list.push_back(dataset_path + "/" + depth_image_name);
    color_image_list.push_back(dataset_path + "/" + color_image_name);
  }
}
void LoadICLTrajectory(std::string dataset_path,
                       std::vector<float4x4> &wTc_list) {
  std::ifstream list_stream(dataset_path + "traj0.gt.freiburg");
  std::string ts_img, img_path, ts_gt;
  float tx, ty, tz, qx, qy, qz, qw;
  while (list_stream >> ts_img
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
    wTc_list.push_back(wTc);
  }
}

void LoadTUMImageList(std::string dataset_path,
                      std::vector<std::string> &depth_image_list,
                      std::vector<std::string> &color_image_list) {
  std::ifstream depth_list_stream(dataset_path + "depth.txt");
  std::ifstream color_list_stream(dataset_path + "rgb.txt");
  std::string time_stamp, file_name;

  std::getline(depth_list_stream, file_name);
  std::getline(depth_list_stream, file_name);
  std::getline(depth_list_stream, file_name);
  while (depth_list_stream >> time_stamp >> file_name) {
    depth_image_list.push_back(dataset_path + "/" + file_name);
  }

  std::getline(color_list_stream, file_name);
  std::getline(color_list_stream, file_name);
  std::getline(color_list_stream, file_name);
  while (color_list_stream >> time_stamp >> file_name) {
    color_image_list.push_back(dataset_path + "/" + file_name);
  }
}
void LoadTUMTrajectory(std::string dataset_path,
                       std::vector<float4x4> &wTc_list) {
  std::ifstream list_stream(dataset_path + "depth_gt_associations.txt");
  std::string ts_img, img_path, ts_gt;
  float tx, ty, tz, qx, qy, qz, qw;
  while (list_stream >> ts_img >> img_path >> ts_gt
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
    wTc_list.push_back(wTc);
  }
}

#endif //VH_CONFIG_READER
