//
// Created by wei on 17-10-24.
//

#include <iomanip>
#include <io/mesh_writer.h>
#include <glog/logging.h>
#include <core/block.h>
#include <core/hash_entry.h>
#include "logging_engine.h"

void LoggingEngine::Init(std::string path) {
  base_path_ = path;

  time_stamp_file_.open(base_path_ + "/mappingstamp.txt");
  if (!time_stamp_file_.is_open()) {
    LOG(ERROR) << "Can't open mappoing stamp file";
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

void LoggingEngine::WriteRawBlocks(const BlockMap &blocks, std::string filename) {
  std::ofstream file(base_path_ + "/Blocks/" + filename + ".block",
                     std::ios::binary);
  if (!file.is_open()) {
    LOG(WARNING) << "can't open block file.";
    return;
  }

  int N = sizeof(std::pair<int3, Block>);
  int num = blocks.size();
  file.write((char *) &num, sizeof(int));
  for (auto &&block:blocks) {
    file.write((char *) &block, N);
  }
  file.close();
}

BlockMap LoggingEngine::ReadRawBlocks(std::string filename) {
  std::ifstream file(base_path_ + "/Blocks/" + filename + ".block");
  BlockMap blocks;
  if (!file.is_open()) {
    LOG(WARNING) << " can't open block file.";
    return blocks;
  }

  int num;
  file.read((char *) &num, sizeof(int));
  if (file.bad()) {
    LOG(WARNING) << " can't open block file.";
    return blocks;
  }

  std::pair<int3, Block> block;
  int N = sizeof(block);
  for (int i = 0; i < num; ++i) {
    file.read((char *) &block, N);
    if (file.bad()) {
      LOG(WARNING) << " did not read the whole block file.";
      return std::move(blocks);
    }
    blocks.insert(block);
  }
  file.close();
  return std::move(blocks);
}

void
LoggingEngine::WriteFormattedBlocks(const BlockMap &blocks, std::string filename) {
  std::ofstream file(base_path_ + "/FormatBlocks/" + filename + ".formatblock");
  if (!file.is_open()) {
    LOG(ERROR) << " can't open format block file.";
    return;
  }

  file << std::left << std::setprecision(3);
  int num = blocks.size();
  file << num << std::endl;
  for (auto &&block:blocks) {
    file << block.first.x << ' ' << block.first.y << ' ' << block.first.z << std::endl;
    int cols = BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH;
    for (int i = 0; i < BLOCK_SIDE_LENGTH; ++i)
      for (int j = 0; j < cols; ++j) {
        file << std::setw(6) << block.second.voxels[i * cols + j].sdf;
        file << (j != cols - 1 ? ' ' : '\n');
      }
    file << std::endl;
    for (int i = 0; i < BLOCK_SIDE_LENGTH; ++i)
      for (int j = 0; j < cols; ++j) {
        file << std::setw(6) << block.second.voxels[i * cols + j].weight;
        file << (j != cols - 1 ? ' ' : '\n');
      }
    file << std::endl;
    file << std::endl;
  }
  file.close();
}

BlockMap LoggingEngine::ReadFormattedBlocks(std::string filename) {
  std::ifstream file(base_path_ + "/FormatBlocks/" + filename + ".formatblock");
  BlockMap blocks;
  if (!file.is_open()) {
    LOG(ERROR) << " can't open format block file.";
    return blocks;
  }

  int num;
  Block block;
  file >> num;
  for (int i = 0; i < num; ++i) {
    int3 pos;
    file >> pos.x >> pos.y >> pos.z;
    int size = BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH;
    block.Clear();
    for (int i = 0; i < size; ++i)
      file >> block.voxels[i].sdf;
    for (int i = 0; i < size; ++i)
      file >> block.voxels[i].weight;
    if (file.bad()) {
      LOG(ERROR) << " can't read the whole format block file.";
      return blocks;
    }
    blocks.emplace(pos, block);
  }
  file.close();
  return blocks;
}

BlockMap LoggingEngine::RecordBlockToMemory(
    const Block *block_gpu, uint block_num,
    const HashEntry *candidate_entry_gpu, uint entry_num
) {

  BlockMap block_map;
  Block *block_cpu = new Block[block_num];
  HashEntry *candidate_entry_cpu = new HashEntry[entry_num];
  cudaMemcpy(block_cpu, block_gpu,
             sizeof(Block) * block_num,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(candidate_entry_cpu, candidate_entry_gpu,
             sizeof(HashEntry) * entry_num,
             cudaMemcpyDeviceToHost);

  for (uint i = 0; i < entry_num; ++i) {
    int3 &pos = candidate_entry_cpu[i].pos;
    CHECK_LT(candidate_entry_cpu[i].ptr, entry_num);
    Block &block = block_cpu[candidate_entry_cpu[i].ptr];
    block_map.emplace(pos, block);
  }

  delete[] block_cpu;
  delete[] candidate_entry_cpu;
  return block_map;
}
