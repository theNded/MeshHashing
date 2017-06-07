#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

// K * c_T_w
uniform mat4 mvp;

out vec3 color_original;

void main() {
  gl_PointSize = 10.0;

  gl_Position = mvp * vec4(position, 1.0);
  color_original = color;
}
