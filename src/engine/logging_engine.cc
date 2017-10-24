//
// Created by wei on 17-10-24.
//

#include <io/mesh_writer.h>
#include "logging_engine.h"

void LoggingEngine::Init(std::string path) {
  base_path_ = path;
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