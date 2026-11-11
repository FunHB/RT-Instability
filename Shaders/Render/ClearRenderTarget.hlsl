cbuffer ClearConfig : register(b0) {
    uint width;
    uint height;
    uint clearColor;
}

RWTexture2D<float4> RenderTarget : register(u0);

float4 UnpackRGBA8(uint packed) {
    return float4(
        (packed & 0xFF)        / 255.0,
        ((packed >> 8) & 0xFF) / 255.0,
        ((packed >> 16) & 0xFF) / 255.0,
        ((packed >> 24) & 0xFF) / 255.0);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;

    if (pixel.x >= width || pixel.y >= height) {
        return;
    }

    RenderTarget[pixel] = UnpackRGBA8(clearColor);
}
