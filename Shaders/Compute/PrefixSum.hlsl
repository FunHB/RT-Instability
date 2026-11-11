cbuffer Config : register(b0) {
    int n;
    uint2 gridSize;
    uint padding;
    float2 gravity;
}

RWStructuredBuffer<uint> CellCounts : register(u0);
RWStructuredBuffer<uint> CellOffsets : register(u1);
RWStructuredBuffer<uint> BlockSums : register(u2);

groupshared uint SharedScan[256];

[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupID : SV_GroupID,
    uint3 groupThreadID : SV_GroupThreadID) {
    uint localIndex = groupThreadID.x;
    uint globalIndex = dispatchThreadID.x;
    uint numberOfCells = gridSize.x * gridSize.y;
    uint value = (globalIndex < numberOfCells) ? CellCounts[globalIndex] : 0;

    SharedScan[localIndex] = value;
    GroupMemoryBarrierWithGroupSync();

    // Blelloch upsweep
    [unroll]
    for (uint strideUp = 1; strideUp < 256; strideUp <<= 1) {
        uint Index = ((localIndex + 1) * strideUp * 2) - 1;
        if (Index < 256) {
            SharedScan[Index] += SharedScan[Index - strideUp];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0) {
        SharedScan[255] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Blelloch downsweep
    [unroll]
    for (uint strideDown = 128; strideDown > 0; strideDown >>= 1) {
        uint Index = ((localIndex + 1) * strideDown * 2) - 1;
        if (Index < 256) {
            uint temp = SharedScan[Index - strideDown];
            SharedScan[Index - strideDown] = SharedScan[Index];
            SharedScan[Index] += temp;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (globalIndex < numberOfCells) {
        CellOffsets[globalIndex] = SharedScan[localIndex];
    }

    if (localIndex == 255) {
        BlockSums[groupID.x] = SharedScan[255] + value;
    }
}