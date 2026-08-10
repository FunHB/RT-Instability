cbuffer ClearConfig : register(b0) {
    uint width;
    uint height;
    uint clearColor;
}

RWTexture2D<uint> RenderTarget : register(u0);

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;

    if (pixel.x >= width || pixel.y >= height) {
        return;
    }

    RenderTarget[pixel] = clearColor;
}
