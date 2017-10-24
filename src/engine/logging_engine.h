//
// Created by wei on 17-10-24.
//

#ifndef MESH_HASHING_LOGGING_ENGINE_H
#define MESH_HASHING_LOGGING_ENGINE_H

#include <string>
#include <opencv2/opencv.hpp>

class LoggingEngine {
public:
  LoggingEngine() = default;
  explicit LoggingEngine(std::string path) : base_path_(path) {};
  void Init(std::string path);

  void ConfigVideoWriter(int width, int height);
  void ConfigPlyWriter();
  void WriteVideo(cv::Mat& mat);
  void WritePly(CompactMesh& mesh);

  bool enable_video() {
    return enable_video_;
  }

  bool enable_ply() {
    return enable_ply_;
  }
private:
  bool enable_video_;
  bool enable_ply_;

  std::string base_path_;
  cv::VideoWriter video_writer_;
};


#endif //MESH_HASHING_LOGGING_ENGINE_H
