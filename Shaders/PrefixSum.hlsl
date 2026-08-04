#define BLOCK_SIZE 1024

cbuffer Cells : register(b0) {
    uint nCells;
}

RWStructuredBuffer<uint> CellOffsets : register(u0);
RWStructuredBuffer<uint> GlobalBlockTracker : register(u1);

groupshared uint sharedData[BLOCK_SIZE];
groupshared uint blockBaseOffset;

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint thid = groupThreadID.x;
    uint globalIdx = groupID.x * BLOCK_SIZE + thid;
    uint offset = 1;

    if (globalIdx < nCells) {
        sharedData[thid] = CellOffsets[globalIdx];
    } else {
        sharedData[thid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // UP-SWEEP (Reduction Phase)
    for (uint depthUp = BLOCK_SIZE >> 1; depthUp > 0; depthUp >>= 1) {
        if (thid < depthUp) {
            uint ai = offset * (2 * thid + 1) - 1;
            uint bi = offset * (2 * thid + 2) - 1;
            sharedData[bi] += sharedData[ai];
        }
        offset *= 2;
        GroupMemoryBarrierWithGroupSync();
    }

    if (thid == 0) {
        uint totalBlockSum = sharedData[BLOCK_SIZE - 1];
        sharedData[BLOCK_SIZE - 1] = 0;

        // Atomically request where this whole block fits into the grand global total scale
        InterlockedAdd(GlobalBlockTracker[0], totalBlockSum, blockBaseOffset);
    }
    GroupMemoryBarrierWithGroupSync();

    // DOWN-SWEEP Phase
    for (uint depthDown = 1; depthDown < BLOCK_SIZE; depthDown *= 2) {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();

        if (thid < depthDown) {
            uint ai = offset * (2 * thid + 1) - 1;
            uint bi = offset * (2 * thid + 2) - 1;

            uint t = sharedData[ai];
            sharedData[ai] = sharedData[bi];
            sharedData[bi] += t;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (globalIdx < nCells) {
        CellOffsets[globalIdx] = sharedData[thid] + blockBaseOffset;
    }

    if (globalIdx == nCells - 1) {
        CellOffsets[nCells] = CellOffsets[nCells - 1] + sharedData[thid];
    }
}