#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

// K * c_T_w
uniform mat4 mvp;
uniform mat4 view_mat;
uniform mat4 model_mat;

out vec3 position_w;
out vec3 normal_c;
out vec3 eye_dir_c;
out vec3 light_dir_c;

void main() {
  gl_PointSize = 10.0;


  gl_Position = mvp * vec4(position, 1.0);
  position_w = (model_mat * vec4(position, 1.0f)).xyz;

  vec3 position_c = (view_mat * model_mat * vec4(position, 1.0f)).xyz;
  eye_dir_c = vec3(0, 0, 0) - position_c;

  vec3 light_w = vec3(0, 2, 3);
  vec3 light_c = (view_mat * vec4(light_w, 1.0f)).xyz;
  light_dir_c = light_c + eye_dir_c;

  normal_c = (view_mat * model_mat * vec4(normal, 0)).xyz;
}
