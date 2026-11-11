cbuffer Config : register(b0) {
    int n;
    uint2 gridSize;
    uint padding;
    float2 gravity;
}

StructuredBuffer<uint> BlockOffsets : register(t0);
RWStructuredBuffer<uint> CellOffsets : register(u0);
RWStructuredBuffer<uint> CellWriteHeads : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint globalIndex = dispatchThreadID.x;
    uint numberOfCells = gridSize.x * gridSize.y;

    if (globalIndex < numberOfCells) {
        uint blockIndex = globalIndex / 256;
        uint offset = CellOffsets[globalIndex] + BlockOffsets[blockIndex];
        CellOffsets[globalIndex] = offset;
        CellWriteHeads[globalIndex] = offset;
    }

    if (globalIndex == 0) {
        CellOffsets[numberOfCells] = (uint)n;
    }
}