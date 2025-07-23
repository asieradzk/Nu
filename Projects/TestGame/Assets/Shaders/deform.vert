#version 330 core

layout (location = 0) in vec2 aPosition;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform float uTime;
uniform float uAmplitude;

out vec2 vTexCoord;

void main()
{
    vec2 position = aPosition;
    
    // Simple sine wave deformation
    position.x += sin(uTime * 2.0 + aPosition.y * 3.0) * uAmplitude;
    
    gl_Position = uProjection * uView * uModel * vec4(position, 0.0, 1.0);
    vTexCoord = aTexCoord;
}