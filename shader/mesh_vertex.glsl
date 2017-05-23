#version 330 core

layout(location = 0) in vec3 position;

uniform mat4 mvp; // K * c_T_w

void main() {
   gl_PointSize = 10.0;
   gl_Position = mvp * vec4(position, 1.0);
}
