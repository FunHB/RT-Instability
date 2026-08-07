cbuffer Config : register(b0) {
    int n;
    float sigma;
    float epsilon;
    float rCut;
    uint2 gridSize;
    float2 g;
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
    for (uint stride = 1; stride < 256; stride <<= 1) {
        uint Index = ((localIndex + 1) * stride * 2) - 1;
        if (Index < 256) {
            SharedScan[Index] += SharedScan[Index - stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0) {
        SharedScan[255] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Blelloch downsweep
    [unroll]
    for (uint stride = 128; stride > 0; stride >>= 1) {
        uint Index = ((localIndex + 1) * stride * 2) - 1;
        if (Index < 256) {
            uint temp = SharedScan[Index - stride];
            SharedScan[Index - stride] = SharedScan[Index];
            SharedScan[Index] += temp;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex < numberOfBlocks) {
        BlockOffsets[localIndex] = SharedScan[localIndex];
    }
}
