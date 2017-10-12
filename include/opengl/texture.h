//
// Created by Neo on 14/08/2017.
//

#ifndef OPENGL_SNIPPET_TEXTURE_H
#define OPENGL_SNIPPET_TEXTURE_H

#include <string>
#include <GL/glew.h>
#include <opencv2/opencv.hpp>

namespace gl {
class Texture {
public:
  /// Texture t = Texture(); t.Init( /* for read or write */);
  Texture() = default;
  ~Texture();

  void Load(std::string texture_path);

  /// Init for reading. Load from file.
  void Init(std::string texture_path);
  /// Init for writing. Note differences in filtering
  void Init(GLint internal_format,
            int width, int height);

  void Bind(int texture_idx);

  const GLuint id() const {
    return texture_id_;
  }
  const int width() const {
    return width_;
  }
  const int height() const {
    return height_;
  }

private:
  cv::Mat texture_;
  int width_;
  int height_;
  GLuint texture_id_;

  bool texture_gened_ = false;

};
}


#endif //OPENGL_SNIPPET_TEXTURE_H
