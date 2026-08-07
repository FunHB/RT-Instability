cbuffer Config : register(b0) {
    int n;
    float sigma;
    float epsilon;
    float rCut;
    uint2 gridSize;
    float2 g;
}

StructuredBuffer<uint> ParticleCellIndices : register(t0);
RWStructuredBuffer<uint> CellWriteHeads : register(u0);
RWStructuredBuffer<uint> SortedParticleIndices : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint Index = dispatchThreadID.x;
    if (Index >= (uint)n) return;

    uint cellIndex = ParticleCellIndices[Index];
    uint writeIndex;
    InterlockedAdd(CellWriteHeads[cellIndex], 1, writeIndex);
    SortedParticleIndices[writeIndex] = Index;
}
