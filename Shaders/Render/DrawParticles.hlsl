struct Particle {
    float mass;
    float2 position;
    float2 velocity;
    float2 force;
    uint id;
    float sigma;
    float epsilon;
};

cbuffer RenderConfig : register(b0) {
    uint n;
    uint width;
    uint height;
    uint pointRadiusPx;
    float4 bounds; // x=minX, y=maxX, z=minY, w=maxY
}

StructuredBuffer<Particle> Particles : register(t0);
StructuredBuffer<uint> Colors : register(t1);

RWTexture2D<float4> RenderTarget : register(u0);

float2 NormalizePosition(float2 position) {
    float2 minBounds = bounds.xz;
    float2 maxBounds = bounds.yw;
    return (position - minBounds) / max(maxBounds - minBounds, 1e-6.xx);
}

int2 ToPixel(float2 normalizedPosition) {
    normalizedPosition = saturate(normalizedPosition);

    // Flip Y for raster coordinates where 0 is top.
    uint x = (uint)(normalizedPosition.x * (float)(width - 1));
    uint y = (uint)((1.0 - normalizedPosition.y) * (float)(height - 1));

    return int2(x, y);
}

float4 UnpackRGBA8(uint packed) {
    return float4(
        (packed & 0xFF)        / 255.0,
        ((packed >> 8) & 0xFF) / 255.0,
        ((packed >> 16) & 0xFF) / 255.0,
        ((packed >> 24) & 0xFF) / 255.0);
}

float4 ColorForId(uint id) {
    if (id < 3) return UnpackRGBA8(Colors[id]);

    return float4(1.0, 1.0, 1.0, 1.0);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= n) {
        return;
    }

    int2 center = ToPixel(NormalizePosition(Particles[index].position));
    int radius = (int)pointRadiusPx;
    float4 color = ColorForId(Particles[index].id);

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if ((x * x + y * y) > (radius * radius)) {
                continue;
            }

            int2 pixel = center + int2(x, y);
            if (pixel.x < 0 || pixel.y < 0 || pixel.x >= (int)width || pixel.y >= (int)height) {
                continue;
            }

            RenderTarget[uint2(pixel)] = color;
        }
    }
}
