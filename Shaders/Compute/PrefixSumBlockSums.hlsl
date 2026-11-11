cbuffer Config : register(b0) {
    int n;
    uint2 gridSize;
    uint padding;
    float2 gravity;
}

RWStructuredBuffer<uint> BlockSums : register(u0);
RWStructuredBuffer<uint> BlockOffsets : register(u1);

groupshared uint SharedScan[256];

[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID) {
    uint localIndex = groupThreadID.x;

    uint numberOfCells = gridSize.x * gridSize.y;
    uint numberOfBlocks = (numberOfCells + 255) / 256;

    uint value = (localIndex < numberOfBlocks) ? BlockSums[localIndex] : 0;
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

    if (localIndex < numberOfBlocks) {
        BlockOffsets[localIndex] = SharedScan[localIndex];
    }
}