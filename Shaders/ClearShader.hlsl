cbuffer ClearConfig : register(b0) {
    uint length;
}

RWStructuredBuffer<uint> Data : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= length) return;

    Data[index] = 0;
}
