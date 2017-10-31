//
// Created by wei on 17-10-24.
//

#ifndef ENGINE_LOGGING_ENGINE_H
#define ENGINE_LOGGING_ENGINE_H

#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

class Int3Sort {
public:
  bool operator()(int3 const &a, int3 const &b) const {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
  }
};
typedef std::map<int3, Block, Int3Sort> BlockMap;

class LoggingEngine {
public:
  LoggingEngine() = default;
  explicit LoggingEngine(std::string path)
      : base_path_(path) {};
  void Init(std::string path);
  ~LoggingEngine();

  void ConfigVideoWriter(int width, int height);
  void ConfigPlyWriter();
  void WriteVideo(cv::Mat& mat);
  void WritePly(CompactMesh& mesh);
  void WriteMappingTimeStamp(double alloc_time, double collect_time, double update_time,
                               int frame_idx);

  BlockMap RecordBlockToMemory(
      const Block *block_gpu, uint block_num,
      const HashEntry *candidate_entry_gpu, uint entry_num
  );
  void WriteFormattedBlocks(const BlockMap &blocks, std::string filename);
  BlockMap ReadFormattedBlocks(std::string filename);
  void WriteRawBlocks(const BlockMap &blocks, std::string filename);
  BlockMap ReadRawBlocks(std::string filename);

  bool enable_video() {
    return enable_video_;
  }
  bool enable_ply() {
    return enable_ply_;
  }
private:
  bool enable_video_ = false;
  bool enable_ply_ = false;

  std::string base_path_;
  std::string prefix_;
  cv::VideoWriter video_writer_;
  std::ofstream time_stamp_file_;
};


#endif //MESH_HASHING_LOGGING_ENGINE_H
