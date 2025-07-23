#version 330 core

layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;

uniform mat4 uMVP;

out vec2 vTexCoord;

void main() {
    // The quad is already in [-0.5, 0.5] space, which will be scaled by transform
    gl_Position = uMVP * vec4(aPosition, 0.0, 1.0);
    vTexCoord = aTexCoord;
}