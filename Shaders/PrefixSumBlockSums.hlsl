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
        uint index = ((localIndex + 1) * stride * 2) - 1;
        if (index < 256) {
            SharedScan[index] += SharedScan[index - stride];
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
        uint index = ((localIndex + 1) * stride * 2) - 1;
        if (index < 256) {
            uint temp = SharedScan[index - stride];
            SharedScan[index - stride] = SharedScan[index];
            SharedScan[index] += temp;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex < numberOfBlocks) {
        BlockOffsets[localIndex] = SharedScan[localIndex];
    }
}
