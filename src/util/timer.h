//
// Created by wei on 17-7-4.
//

#ifndef VOXEL_HASHING_TIMER_H
#define VOXEL_HASHING_TIMER_H

#include <chrono>

class Timer {
private:
  std::chrono::time_point<std::chrono::system_clock> start_, end_;

public:
  void Tick() {
    start_ = std::chrono::system_clock::now();
  }
  double Tock() {
    end_ = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = end_ - start_;
    return seconds.count();
  }
};
#endif //VOXEL_HASHING_TIMER_H
