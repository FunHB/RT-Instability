cbuffer Config : register(b0) {
    uint groupCount;
}

StructuredBuffer<float> PartialEnergy : register(t0);
RWStructuredBuffer<float> TotalEnergy : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {
    if (threadID.x != 0) return;

    float sum = 0.0;

    for (uint i = 0; i < groupCount; ++i) {
        sum += PartialEnergy[i];
    }

    TotalEnergy[0] = sum;
}
