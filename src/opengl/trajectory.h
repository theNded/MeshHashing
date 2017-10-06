//
// Created by Neo on 23/08/2017.
//

#ifndef OPENGL_SNIPPET_TRAJECTORY_H
#define OPENGL_SNIPPET_TRAJECTORY_H

#include <fstream>
#include <vector>
#include <glm/glm.hpp>

namespace gl {
class Trajectory {
public:
  Trajectory() = default;

  std::vector<glm::mat4>& poses() {
    return poses_;
  }
  void Load(std::string path);
  void Save(std::string path);

private:
  std::vector<glm::mat4> poses_;
};
}


#endif //OPENGL_SNIPPET_TRAJECTORY_H
