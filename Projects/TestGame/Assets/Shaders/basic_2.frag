#version 330 core
out vec4 FragColor;

uniform vec4 uColor;
uniform float uTime;
uniform float uSpeed;

void main()
{
    // HARDCODED BLACK FOR TESTING
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}