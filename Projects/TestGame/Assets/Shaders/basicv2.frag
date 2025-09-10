#version 330 core
out vec4 FragColor;

uniform vec4 uColor;
uniform float uTime;
uniform float uSpeed;

void main()
{
    // Rainbow effect based on time
    if (uTime == 0.0) {
        // When time is 0, use the input color directly
        FragColor = uColor;
    } else {
        // Create rainbow effect using sine waves offset by 120 degrees
        float t = mod(uTime * uSpeed, 6.28318); // Keep time in 0 to 2*PI range
        
        float r = sin(t) * 0.5 + 0.5;
        float g = sin(t + 2.0944) * 0.5 + 0.5; // +120 degrees
        float b = sin(t + 4.1888) * 0.5 + 0.5; // +240 degrees
        
        // Mix rainbow with base color (allows tinting the rainbow)
        vec3 rainbow = vec3(r, g, b);
        vec3 finalColor = rainbow * uColor.rgb;
        
        FragColor = vec4(finalColor, uColor.a);
    }
}