#version 330 core

layout(location = 0) in vec2 position;

out vec2 uv;

void main() {
    gl_Position = vec4(position, 0, 1);

    // (-1, 1) -> (0, 1)
    uv = 0.5 * position + vec2(0.5, 0.5);

    // origin for vertex  : left bottom
    //        for texture : left top
    uv = vec2(uv.s, 1 - uv.t);
}