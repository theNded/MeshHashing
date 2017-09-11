//
// Created by Neo on 23/08/2017.
//

#include "trajectory.h"

namespace gl {
void Trajectory::Load(std::string path) {
  std::ifstream in(path);
  int n;
  in >> n;
  poses_.clear();
  poses_.reserve(n);

  glm::mat4 pose;
  while (n--) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j)
        in >> pose[i][j];
    }
    poses_.push_back(pose);
  }
}

void Trajectory::Save(std::string path) {
  std::ofstream out(path);
  out << poses_.size() << std::endl;
  for (auto &pose : poses_) {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        out << pose[i][j] << " ";
    out << std::endl;
  }
}
}