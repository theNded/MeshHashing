/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include <glog/logging.h>

#include "params.h"
#include "config_reader.h"
#include "map.h"
#include "sensor.h"
#include "ray_caster.h"

using namespace std;

extern void SetConstantSDFParams(const SDFParams& params);

void LoadImages(const string &strAssociationFilename,
                vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD,
                vector<double> &vTimestamps) {
  ifstream fAssociation;
  LOG(INFO) << strAssociationFilename.c_str();
  fAssociation.open(strAssociationFilename.c_str());
  while (!fAssociation.eof()) {
    string s;
    getline(fAssociation, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      LOG(INFO) << ss.str();
      double t;
      string sRGB, sD;
      ss >> t;
      vTimestamps.push_back(t);
      ss >> sRGB;
      vstrImageFilenamesRGB.push_back(sRGB);
      ss >> t;
      ss >> sD;
      vstrImageFilenamesD.push_back(sD);
    }
  }
}

float4x4 MatTofloat4x4(cv::Mat m) {
  float4x4 T;
  T.setIdentity();
  T.m11 = (float)m.at<float>(0,0);
  T.m12 = (float)m.at<float>(0,1);
  T.m13 = (float)m.at<float>(0,2);
  T.m14 = (float)m.at<float>(0,3);
  T.m21 = (float)m.at<float>(1,0);
  T.m22 = (float)m.at<float>(1,1);
  T.m23 = (float)m.at<float>(1,2);
  T.m24 = (float)m.at<float>(1,3);
  T.m31 = (float)m.at<float>(2,0);
  T.m32 = (float)m.at<float>(2,1);
  T.m33 = (float)m.at<float>(2,2);
  T.m34 = (float)m.at<float>(2,3);

  LOG(INFO) << "------------------";
  LOG(INFO) << T.m11 << " " << T.m12 << " " << T.m13 << " " << T.m14;
  LOG(INFO) << T.m21 << " " << T.m22 << " " << T.m23 << " " << T.m24;
  LOG(INFO) << T.m31 << " " << T.m32 << " " << T.m33 << " " << T.m34;
  LOG(INFO) << T.m41 << " " << T.m42 << " " << T.m43 << " " << T.m44;

  return T;
}

int main(int argc, char **argv) {
  // Retrieve paths to images
  vector<string> vstrImageFilenamesRGB;
  vector<string> vstrImageFilenamesD;
  vector<double> vTimestamps;
  LOG(INFO) << "Loading image list";

  string strAssociationFilename =
          "/home/wei/data/TUM/rgbd_dataset_freiburg1_xyz"
          "/rgb_depth_association.txt";
  string path_to_vocabulary = "/home/wei/softwares/ORB_SLAM2/Vocabulary/ORBvoc.txt";
  string path_to_config = "/home/wei/softwares/ORB_SLAM2/Examples/RGB-D/TUM1.yaml";
  string dataset_path = "/home/wei/data/TUM/rgbd_dataset_freiburg1_xyz";

  LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD,
             vTimestamps);

  // Check consistency in the number of images and depthmaps
  int nImages = vstrImageFilenamesRGB.size();
  if (vstrImageFilenamesRGB.empty()) {
    cerr << endl << "No images found in provided path." << endl;
    return 1;
  } else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size()) {
    cerr << endl << "Different number of images for rgb and depth." << endl;
    return 1;
  }

  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  LOG(INFO) << "Loading vocabulary";
  ORB_SLAM2::System SLAM(path_to_vocabulary, path_to_config,
                         ORB_SLAM2::System::RGBD, true);
  LOG(INFO) << "Loading vocabulary finished";
  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  cout << endl << "-------" << endl;
  cout << "Start processing sequence ..." << endl;
  cout << "Images in the sequence: " << nImages << endl << endl;


  ConfigReader config;
  config.LoadConfig("../config/tum1.yml");
  SetConstantSDFParams(config.sdf_params);

  Map voxel_map(config.hash_params);
  LOG(INFO) << "map allocated";

  Sensor sensor(config.sensor_params);
  sensor.BindGPUTexture();

  RayCaster ray_caster(config.ray_caster_params);

  // Main loop
  cv::Mat imRGB, imD;
  for (int ni = 0; ni < nImages; ni++) {
    // Read image and depthmap from file
    imRGB = cv::imread(string(dataset_path) + "/" + vstrImageFilenamesRGB[ni],
                       CV_LOAD_IMAGE_UNCHANGED);
    imD = cv::imread(string(dataset_path) + "/" + vstrImageFilenamesD[ni],
                     CV_LOAD_IMAGE_UNCHANGED);
    double tframe = vTimestamps[ni];

    if (imRGB.empty()) {
      cerr << endl << "Failed to load image at: "
           << string(dataset_path) << "/" << vstrImageFilenamesRGB[ni] << endl;
      return 1;
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // Pass the image to the SLAM system
    cv::Mat Tcw = SLAM.TrackRGBD(imRGB, imD, tframe);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(
            t2 - t1).count();
    vTimesTrack[ni] = ttrack;

    /// Voxel hashing
    cv::cvtColor(imRGB, imRGB, CV_BGR2BGRA);

    sensor.Process(imD, imRGB);
    float4x4 cTw = MatTofloat4x4(Tcw);
    cTw = cTw.getInverse();
    sensor.set_transform(cTw);

    voxel_map.Integrate(sensor, NULL);
    voxel_map.MarchingCubes();
    voxel_map.CompressMesh();

    if (ni > 0 && ni % 500 == 0) {
      std::stringstream ss;
      ss.str("");
      ss << "test" << "_" << ni << ".obj";
      voxel_map.SaveMesh(ss.str());
    }

    ray_caster.Cast(voxel_map, cTw.getInverse());
    cv::imshow("normal", ray_caster.normal_image());
    cv::waitKey(1);
  }

  // Stop all threads
  SLAM.Shutdown();

  // Tracking time statistics
  sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = 0;
  for (int ni = 0; ni < nImages; ni++) {
    totaltime += vTimesTrack[ni];
  }
  cout << "-------" << endl << endl;
  cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
  cout << "mean tracking time: " << totaltime / nImages << endl;

//    // Save camera trajectory
//    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
//    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  return 0;
}