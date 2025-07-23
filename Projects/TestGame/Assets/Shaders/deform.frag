#version 330 core

in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uMainTexture;

void main()
{
    FragColor = texture(uMainTexture, vTexCoord);
}