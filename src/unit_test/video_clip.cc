//
// Created by wei on 17-7-11.
//

#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

int main() {
  cv::VideoCapture reader;
  reader.open("/home/wei/Documents/incremental/tum-color.avi");

  cv::Mat frame;
  int i = 0;
  while (1) {
    std::cout << i++ << std::endl;
    reader >> frame;

    if (i == 100 || i == 400 || i == 800 || i == 1400 || i == 2450) {
      std::stringstream ss;
      ss << "tum-color-" << i << ".png";
      cv::imwrite(ss.str(), frame);
    }
    cv::imshow("video", frame);
    if (cv::waitKey(10) == 27) break;
  }
}