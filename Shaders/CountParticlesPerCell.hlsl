struct Particle {
    float mass;
    float2 position;
    float2 velocity;
    float2 force;
    uint2 cell;
    uint id;
};

cbuffer Config : register(b0) {
    int n;
    float sigma;
    float epsilon;
    float rCut;
    int2 gridSize;
}

StructuredBuffer<Particle> Particles : register(t0);

RWStructuredBuffer<uint> CellOffsets : register(u0);

uint GetCellIndex(uint2 cell) {
    return cell.y + cell.x * gridSize.y;
}

[numthreads(256, 1, 1)] 
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint Index = dispatchThreadID.x;

    if (Index >= (uint)n) return;

    uint cellIndex = GetCellIndex(Particles[Index].cell);

    InterlockedAdd(CellOffsets[cellIndex], 1);
}