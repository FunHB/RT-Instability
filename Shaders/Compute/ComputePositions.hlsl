struct Particle {
    float mass;
    float2 position;
    float2 velocity;
    float2 force;
    uint id;
    float sigma;
    float epsilon;
};

struct TimeData {
    uint frame;
    float dt;
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

#define DesiredEnergy 60 * n 

RWStructuredBuffer<Particle> Particles : register(u0);

StructuredBuffer<TimeData> Time : register(t0);

StructuredBuffer<float> Energy : register(t1);

void ComputeX(uint i) {
    float a = Time[0].dt * 0.5 / Particles[i].mass;
    Particles[i].position += Time[0].dt * (Particles[i].velocity + a * Particles[i].force);
}

void ComputeV(uint i) {
    float a = Time[0].dt * 0.5 / Particles[i].mass;
    Particles[i].velocity += a * Particles[i].force;
}

void ApplyPeriodicBoundary(inout float2 position) {
    float width = bounds.y - bounds.x;
    position.x = bounds.x + fmod(position.x - bounds.x + width, width);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint Index = dispatchThreadID.x;

    if (Index >= (uint)n) return;

    ComputeX(Index);
    ApplyPeriodicBoundary(Particles[Index].position);
    ComputeV(Index);

    if (Time[0].frame % 1000 == 0) {
        float totalEnergy = Energy[0];
        // Particles[Index].velocity *= sqrt(DesiredEnergy / max(totalEnergy, 1e-20));
    }
}
