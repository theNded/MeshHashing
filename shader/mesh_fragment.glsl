#version 330 core

in vec3 normal_out;
in vec3 position_out;

out vec3 color;

void main() {
	float g = abs(dot(normal_out, normalize(vec3(0, 2, 0) - position_out)));
	color = vec3(g, g, g);
}