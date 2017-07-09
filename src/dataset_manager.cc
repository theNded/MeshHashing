//
// Created by wei on 17-5-31.
//

#include "dataset_manager.h"

#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

void LoadRuntimeParams(std::string path, RuntimeParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.dataset_type  = (int)fs["dataset_type"];

  params.free_walk     = (int)fs["free_walk"];
  params.ploygon_mode     = (int)fs["ploygon_mode"];
  params.mesh_range = (int)fs["mesh_range"];
  params.fine_gradient = (int)fs["fine_gradient"];
  params.render_type   = (int)fs["render_type"];

  params.bounding_box  = (int)fs["bounding_box"];
  params.ray_casting   = (int)fs["ray_casting"];

  params.record_video  = (int)fs["record_video"];
  params.save_mesh     = (int)fs["save_mesh"];
  params.filename_prefix = (std::string)fs["filename_prefix"];
  params.time_profile    = (std::string)fs["time_profile"];
  params.memo_profile    = (std::string)fs["memo_profile"];

  params.run_frames    = (int)fs["run_frames"];
}

void LoadHashParams(std::string path, HashParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.bucket_count     = (int)fs["bucket_count"];
  params.bucket_size      = (int)fs["bucket_size"];
  params.entry_count      = (int)fs["entry_count"];
  params.linked_list_size = (int)fs["linked_list_size"];
  params.value_capacity   = (int)fs["value_capacity"];
}

void LoadMeshParams(std::string path, MeshParams &params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.max_vertex_count   = (int)fs["max_vertex_count"];
  params.max_triangle_count = (int)fs["max_triangle_count"];
}

void LoadSDFParams(std::string path, SDFParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.voxel_size                = (float)fs["voxel_size"];
  params.sdf_upper_bound           = (float)fs["sdf_upper_bound"];
  params.truncation_distance_scale = (float)fs["truncation_distance_scale"];
  params.truncation_distance       = (float)fs["truncation_distance"];
  params.weight_sample             = (int)fs["weight_sample"];
  params.weight_upper_bound        = (int)fs["weight_upper_bound"];
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

void LoadRayCasterParams(std::string path, RayCasterParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.min_raycast_depth    = (float)fs["min_raycast_depth"];
  params.max_raycast_depth    = (float)fs["max_raycast_depth"];
  params.raycast_step         = (float)fs["raycast_step"];
  params.sample_sdf_threshold = (float)fs["sample_sdf_threshold"];
  params.sdf_threshold        = (float)fs["sdf_threshold"];
  params.enable_gradients     = (int)fs["enable_gradient"];
}


/// 1-1-1 correspondences
void LoadICL(std::string               dataset_path,
             std::vector<std::string> &depth_image_list,
             std::vector<std::string> &color_image_list,
             std::vector<float4x4>& wTcs) {
  std::ifstream img_stream(dataset_path + "associations.txt");
  std::string time_stamp, depth_image_name, color_image_name;
  /// !!! ICL problem: pose of the 1st frame is not provided
  img_stream >> time_stamp >> depth_image_name
             >> time_stamp >> color_image_name;
  while (img_stream >> time_stamp >> depth_image_name
                    >> time_stamp >> color_image_name) {
    depth_image_list.push_back(dataset_path + "/" + depth_image_name);
    color_image_list.push_back(dataset_path + "/" + color_image_name);
  }

  std::ifstream traj_stream(dataset_path + "traj0.gt.freiburg");
  std::string ts_img, img_path, ts_gt;
  float tx, ty, tz, qx, qy, qz, qw;
  float4x4 rTl;
  rTl.setIdentity();
  rTl.entries2[1][1] = -1;

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

    wTc = rTl * wTc * rTl.getInverse();
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

void LoadSUN3DOriginal(std::string dataset_path,
                       std::vector<std::string> &depth_img_list,
                       std::vector<std::string> &color_img_list,
                       std::vector<float4x4> &wTcs) {
  std::ifstream stream(dataset_path + "image_depth_association.txt");
  std::string ts, color_img_name, depth_img_name;
  while (stream >> ts >> color_img_name >> ts >> depth_img_name) {
    color_img_list.push_back(dataset_path + "image/" + color_img_name);
    depth_img_list.push_back(dataset_path + "depth/" + depth_img_name);
  }

  std::ifstream traj_stream(dataset_path + "trajectory.txt");
  double cTw[12];
  float4x4 wTc;
  while (traj_stream >> cTw[0] >> cTw[1] >> cTw[2] >> cTw[3]
                     >> cTw[4] >> cTw[5] >> cTw[6] >> cTw[7]
                     >> cTw[8] >> cTw[9] >> cTw[10] >> cTw[11]) {

    wTc.setIdentity();
    wTc.m11 = (float)cTw[0];
    wTc.m12 = (float)cTw[1];
    wTc.m13 = (float)cTw[2];
    wTc.m14 = (float)cTw[3];
    wTc.m21 = (float)cTw[4];
    wTc.m22 = (float)cTw[5];
    wTc.m23 = (float)cTw[6];
    wTc.m24 = (float)cTw[7];
    wTc.m31 = (float)cTw[8];
    wTc.m32 = (float)cTw[9];
    wTc.m33 = (float)cTw[10];
    wTc.m34 = (float)cTw[11];

    wTc.getInverse();
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

void Load3DVCR(std::string dataset_path,
               std::vector<std::string> &depth_image_list,
               std::vector<std::string> &color_image_list,
               std::vector<float4x4>& wTcs) {

  std::ifstream traj_stream(dataset_path + "trajectory.txt");
  std::string ts_img, img_path, ts_gt;
  float ts, tx, ty, tz, qx, qy, qz, qw;

  std::unordered_set<int> tracked_ts;
  while (traj_stream >> ts
                     >> tx >> ty >> tz
                     >> qx >> qy >> qz >> qw) {
    tracked_ts.emplace((int)ts);
    LOG(INFO) << (int)ts;

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

  std::ifstream color_stream(dataset_path + "rgb.txt");
  std::string img_name;

  int count = 0;
  while (color_stream >> img_name) {
    if (tracked_ts.find(count) != tracked_ts.end()) {
      LOG(INFO) << dataset_path + "rgb/" + img_name;
      color_image_list.push_back(dataset_path + "rgb/" + img_name);
    }
    ++count;
  }

  count = 0;
  std::ifstream depth_stream(dataset_path + "depth.txt");
  while (depth_stream >> img_name) {
    if (tracked_ts.find(count) != tracked_ts.end()) {
      LOG(INFO) << dataset_path + "depth/" + img_name;
      depth_image_list.push_back(dataset_path + "depth/" + img_name);
    }
    ++count;
  }
}

const std::string kConfigPaths[] = {
    "../config/ICL.yml",
    "../config/TUM1.yml",
    "../config/TUM2.yml",
    "../config/TUM3.yml",
    "../config/SUN3D.yml",
    "../config/SUN3D_ORIGINAL.yml",
    "../config/PKU.yml"
};

////////////////////
/// class ConfigManager
////////////////////
void ConfigManager::LoadConfig(std::string config_path) {
  LoadHashParams(config_path, hash_params);
  LoadMeshParams(config_path, mesh_params);
  LoadSDFParams(config_path, sdf_params);
  LoadSensorParams(config_path, sensor_params);
  LoadRayCasterParams(config_path, ray_caster_params);

  ray_caster_params.width = sensor_params.width;
  ray_caster_params.height = sensor_params.height;
  ray_caster_params.fx = sensor_params.fx;
  ray_caster_params.fy = sensor_params.fy;
  ray_caster_params.cx = sensor_params.cx;
  ray_caster_params.cy = sensor_params.cy;
}

void ConfigManager::LoadConfig(DatasetType dataset_type) {
  std::string config_path = kConfigPaths[dataset_type];
  LoadConfig(config_path);
}

////////////////////
/// class DataManager
////////////////////
void DataManager::LoadDataset(DatasetType dataset_type) {
  std::string config_path = kConfigPaths[dataset_type];
  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  std::string dataset_path = (std::string)fs["dataset_path"];
  LoadDataset(dataset_path, dataset_type);
}

void DataManager::LoadDataset(Dataset dataset) {
  LoadDataset(dataset.path, dataset.type);
}

void DataManager::LoadDataset(std::string dataset_path,
                              DatasetType dataset_type) {
  switch (dataset_type) {
    case ICL:
      LoadICL(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case SUN3D:
      LoadSUN3D(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case SUN3D_ORIGINAL:
      LoadSUN3DOriginal(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case TUM1:
      LoadTUM(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case TUM2:
      LoadTUM(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case TUM3:
      LoadTUM(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case PKU:
      Load3DVCR(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
  }
}

bool DataManager::ProvideData(cv::Mat &depth,
                              cv::Mat &color) {
  if (frame_id > depth_image_list.size()) {
    LOG(ERROR) << "All images provided!";
    return false;
  }
  depth = cv::imread(depth_image_list[frame_id], CV_LOAD_IMAGE_UNCHANGED);
  color = cv::imread(color_image_list[frame_id]);
  if (color.channels() == 3) {
    cv::cvtColor(color, color, CV_BGR2BGRA);
  }
  ++frame_id;

  return true;
  // TODO: Network situation
}

bool DataManager::ProvideData(cv::Mat &depth,
                              cv::Mat &color,
                              float4x4 &wTc) {
  if (frame_id >= depth_image_list.size()) {
    LOG(ERROR) << "All images provided!";
    return false;
  }
  LOG(INFO) << frame_id << "/" << depth_image_list.size();
  depth = cv::imread(depth_image_list[frame_id], CV_LOAD_IMAGE_UNCHANGED);
  color = cv::imread(color_image_list[frame_id]);
  if (color.channels() == 3) {
    cv::cvtColor(color, color, CV_BGR2BGRA);
  }

  wTc   = wTcs[0].getInverse() * wTcs[frame_id];
  ++frame_id;

  return true;
  // TODO: Network situation
}