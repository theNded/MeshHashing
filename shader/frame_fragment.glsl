#version 330 core

in vec2 uv;
uniform sampler2D texture_sampler;

out vec4 color;

void main() {
    color = texture(texture_sampler, uv);
}