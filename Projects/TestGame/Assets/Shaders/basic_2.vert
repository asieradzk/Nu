#version 330
layout(location = 0) in vec2 aPosition;

uniform mat4 uTransform;
uniform mat4 uViewProjection;

void main() {
    // Transform vertex position using model-view-projection matrix
    vec4 worldPos = uTransform * vec4(aPosition, 0.0, 1.0) * 2;
    gl_Position = uViewProjection * worldPos;
}