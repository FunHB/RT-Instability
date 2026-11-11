struct Particle {
    float mass;
    float2 position;
    float2 velocity;
    float2 force;
    uint id;
    float sigma;
    float epsilon;
};

cbuffer Config : register(b0) {
    int n;
    uint2 gridSize;
    uint padding;
    float2 gravity;
}

cbuffer Boundaries : register(b1) {
    float4 bounds; // x=minX, y=maxX, z=minY, w=maxY
}

StructuredBuffer<Particle> Particles : register(t0);
RWStructuredBuffer<uint> CellCounts : register(u0);
RWStructuredBuffer<uint> ParticleCellIndices : register(u1);

int2 GetCell(float2 position) {
    float2 normalizedPos = (position - bounds.xz) * gridSize / (bounds.yw - bounds.xz);
    return clamp((int2)normalizedPos, int2(0, 0), int2(gridSize.xy - 1));
}

uint GetCellIndex(uint2 cell) {
    return cell.x + cell.y * gridSize.x;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint Index = dispatchThreadID.x;
    if (Index >= (uint)n) return;

    int2 cell = GetCell(Particles[Index].position);
    uint cellIndex = GetCellIndex(uint2(cell));
    uint previousCount;

    ParticleCellIndices[Index] = cellIndex;
    InterlockedAdd(CellCounts[cellIndex], 1, previousCount);
}
