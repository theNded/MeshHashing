#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 mvp; // K * c_T_w

out vec3 normal_frag;
out vec3 position_frag;

void main() {
   gl_PointSize = 10.0;
   gl_Position = mvp * vec4(position, 1.0);

   normal_frag   = normal;
   position_frag = position;
}
