cbuffer Config : register(b0) {
    int n;
    float sigma;
    float epsilon;
    float rCut;
    int2 gridSize;
}

RWStructuredBuffer<uint> CellOffsets : register(u0);
RWStructuredBuffer<uint> GlobalBlockTracker : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    // Reset the global block tracker to 0 for the first thread
    if (dispatchThreadID.x == 0) {
        GlobalBlockTracker[0] = 0;
    }

    // Only clear up to the number of cells
    if (dispatchThreadID.x < (uint)(gridSize.x * gridSize.y)) {
        CellOffsets[dispatchThreadID.x] = 0;
    }
}