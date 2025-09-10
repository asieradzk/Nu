#shader vertex
#version 330
layout(location = 0) in vec2 aPosition;

uniform mat4 uTransform;
uniform mat4 uViewProjection;

void main() {
    // Transform vertex position using model-view-projection matrix
    vec4 worldPos = uTransform * vec4(aPosition, 0.0, 1.0);
    gl_Position = uViewProjection * worldPos;
}

#shader fragment
// Assets/Shaders/basic.frag
#version 330 core
out vec4 FragColor;

uniform vec4 uColor;

void main()
{
    FragColor = uColor;
}