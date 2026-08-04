struct Particle {
    float mass;
    float2 position;
    float2 velocity;
    float2 force;
    uint2 cell;
    uint id;
};

struct TimeData {
    float totalTime;
    float dt;
};

cbuffer Config : register(b0) {
    int n;
    float sigma;
    float epsilon;
    float rCut;
    int2 gridSize;
}

cbuffer Boundaries : register(b1) {
    float4 bounds; // x=minX, y=maxX, z=minY, w=maxY
}

RWStructuredBuffer<Particle> Particles : register(u0);
RWStructuredBuffer<float2> Positions : register(u1);

StructuredBuffer<TimeData> SimulationTime : register(t0);

float map(float input, float inputStart, float inputEnd, float outputStart, float outputEnd) {
    float slope = 1.0 * (outputEnd - outputStart) / (inputEnd - inputStart);
    return outputStart + slope * (input - inputStart);
}

float2 map(float2 input, float2 inputStart, float2 inputEnd, float2 outputStart, float2 outputEnd) {
    return float2(
        map(input.x, inputStart.x, inputEnd.x, outputStart.x, outputEnd.x), 
        map(input.y, inputStart.y, inputEnd.y, outputStart.y, outputEnd.y));
}

void AssignCell(uint i) {
    float2 normalizedPos = map(Particles[i].position, bounds.xz, bounds.yw, 0.xx, float2(gridSize.xy));
    Particles[i].cell = (uint2)clamp(floor(normalizedPos), 0.xx, float2(gridSize.xy - 1));
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint Index = dispatchThreadID.x;

    if (Index >= (uint)n) return;

    float2 a = Particles[Index].force / Particles[Index].mass;
    float dt = SimulationTime[0].dt;

    Particles[Index].position += dt * Particles[Index].velocity + 0.5 * dt * dt * a;
    Particles[Index].velocity += .5 * SimulationTime[0].dt * a;

    Positions[Index] = (float2)Particles[Index].position;

    AssignCell(Index);
}