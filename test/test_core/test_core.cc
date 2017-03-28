#include <glog/logging.h>

#include "test_core.h"

int main() {
  TestCore test;
  Voxel a = test.Run();

  LOG(INFO) << a.sdf;
  LOG(INFO) << (int)a.color.x << (int)a.color.y << (int)a.color.z;
  LOG(INFO) << (int)a.weight;

  return 0;
}