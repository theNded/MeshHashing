#version 330 core

#define LIGHT_COUNT 1

// Interpolated values from the vertex shaders
in vec3 position_c;
in vec3 normal_c;
in vec3 light_c[LIGHT_COUNT];
uniform float light_power;
uniform vec3 light_color;

// Ouput data
out vec3 color;

// Values that stay constant for the whole mesh.
uniform sampler2D texture_sampler;

void main(){
    vec3 diffuse_color  = vec3(0.88f, 0.72f, 0.62f);
	vec3 ambient_color  = vec3(0.2, 0.2, 0.2) * diffuse_color;
	vec3 specular_color = vec3(0.3, 0.3, 0.3);

    color = vec3(0, 0, 0);

    vec3 lambertian = vec3(0);
    vec3 specular   = vec3(0);
	for (int i = 0; i < LIGHT_COUNT; ++i) {
        float distance = length(light_c[i] - position_c);

        vec3 n = normalize(normal_c);
        vec3 l = normalize(light_c[i] - position_c);
        float cos_theta = clamp(dot(n, l), 0, 1);

        vec3 e = -normalize(position_c);
        vec3 r = reflect(-l, n);
        float cos_alpha = clamp(dot(e, r), 0, 1);

        vec3 factor = light_color * light_power / (distance * distance);
        lambertian += cos_theta * factor;
        specular   += cos_alpha * pow(cos_alpha, 5) * factor;
	}

	lambertian = clamp(lambertian, 0, 1);
	specular = clamp(specular, 0, 1);

    color =
        // Ambient : simulates indirect lighting
         ambient_color +
        // Diffuse : "color" of the object
        diffuse_color * lambertian +
        // Specular : reflective highlight, like a mirror
        specular_color * specular;
}