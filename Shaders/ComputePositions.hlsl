struct Particle {
    float mass;
    float2 position;
    float2 velocity;
    float2 force;
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

RWStructuredBuffer<TimeData> SimulationTime : register(u2);

void ComputeX(uint i) {
    float a = SimulationTime[0].dt * 0.5 / Particles[i].mass;
    Particles[i].position += SimulationTime[0].dt * (Particles[i].velocity + a * Particles[i].force);
}

void ComputeV(uint i) {
    float a = SimulationTime[0].dt * 0.5 / Particles[i].mass;
    Particles[i].velocity += a * Particles[i].force;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint Index = dispatchThreadID.x;

    if (Index == 0) {
        SimulationTime[0].totalTime += SimulationTime[0].dt;
    }

    if (Index >= (uint)n) return;

    ComputeX(Index);
    ComputeV(Index);

    Positions[Index] = Particles[Index].position;
}