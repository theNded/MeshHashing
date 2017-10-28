//
// Created by wei on 17-10-24.
//

#include <iomanip>
#include <io/mesh_writer.h>
#include <glog/logging.h>
#include "logging_engine.h"

void LoggingEngine::Init(std::string path) {
  base_path_ = path;

  time_stamp_file_.open(base_path_ + "/mappingstamp.txt");
  if (!time_stamp_file_.is_open()) {
    LOG(ERROR) << "Can't open mappoint stamp file";
    return;
  }

  time_stamp_file_.flags(std::ios::right);
  time_stamp_file_.setf(std::ios::fixed);
}

LoggingEngine::~LoggingEngine() {
  if (video_writer_.isOpened())
    video_writer_.release();
  if (time_stamp_file_.is_open())
    time_stamp_file_.close();
  time_stamp_file_<<std::setprecision(4);
}

void LoggingEngine::ConfigVideoWriter(int width, int height) {
  enable_video_ = true;
  video_writer_.open(base_path_ + "/video.avi",
                     CV_FOURCC('X', 'V', 'I', 'D'),
                     30, cv::Size(width, height));
}

void LoggingEngine::WriteVideo(cv::Mat &mat) {
  video_writer_ << mat;
}

void LoggingEngine::ConfigPlyWriter() {
  enable_ply_ = true;
}
void LoggingEngine::WritePly(CompactMesh &mesh) {
  SavePly(mesh, base_path_ + "/mesh.ply");
}

void LoggingEngine::WriteMappingTimeStamp(double alloc_time,
                                          double collect_time,
                                          double update_time,
                                          int frame_idx) {

  time_stamp_file_ << "For " << frame_idx << "-th frame, "
                   << "alloc time : " << alloc_time * 1000 << "ms "
                   << "collect time : " << collect_time * 1000 << "ms "
                   << "update time : " << update_time * 1000 << "ms\n";
}
