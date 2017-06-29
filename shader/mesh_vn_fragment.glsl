#version 330 core

in vec3 position_w;
in vec3 normal_c;
in vec3 eye_dir_c;
in vec3 light_dir_c;

out vec3 color;

void main() {
  vec3  light_w = vec3(0, 3, 0);
	vec3  light_color = vec3(1, 1, 1);
	float light_power = 10.0f;

  /// Diffuse x light
  vec3 diffuse_color = vec3(0.88f, 0.72f, 0.62f);

	vec3 n = normalize(normal_c);
	vec3 l = normalize(light_dir_c);
	float cos_theta = clamp(dot(n, l), 0, 1);
	float distance = length(light_w - position_w);

	color = diffuse_color * light_color * light_power
	* cos_theta / (distance * distance);

  /// Ambient
  vec3 ambient_color = vec3(0.1f, 0.1f, 0.1f) * diffuse_color;
  color += ambient_color;

  /// Specular part
	vec3 specular_color = vec3(0.3, 0.3, 0.3);

  vec3 e = normalize(eye_dir_c);
  vec3 r = reflect(-l, n);
  float cos_alpha = clamp(dot(e, r), 0, 1);

	color += specular_color * light_color * light_power
	 * pow(cos_alpha, 5) / (distance * distance);
}