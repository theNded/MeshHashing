#version 330 core

out vec3 color;
uniform vec3 uni_color;

void main() {
  color = uni_color;
}