#version 330
layout(location = 0) in vec2 aPosition;

uniform vec3 uPosition;
uniform vec2 uSize;
uniform mat4 uViewProjection; // You'll need to pass this too

void main() {
    // Transform vertex from [-1,1] quad space to world space
    vec2 worldPos = (aPosition * 0.5 + 0.5) * uSize + uPosition.xy;
    
    // For now, just normalize to screen space (-1 to 1)
    // Later you'll multiply by view projection matrix
    gl_Position = vec4(worldPos / 400.0 - 1.0, 0.0, 1.0); // Assuming 800x600 viewport
}