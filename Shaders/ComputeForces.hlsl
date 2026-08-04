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
    uint2 gridSize;
    float2 g;
}

cbuffer Boundaries : register(b1) {
    float4 bounds; // x=minX, y=maxX, z=minY, w=maxY
}

RWStructuredBuffer<Particle> Particles : register(u0);

StructuredBuffer<uint> CellOffsets : register(t0);
StructuredBuffer<uint> SortedParticleIndices : register(t1);
StructuredBuffer<TimeData> SimulationTime : register(t2);

uint GetCellIndex(uint2 cell) {
    return cell.y + cell.x * gridSize.y;
}

void Force(uint i, uint j) {
    float2 rij = Particles[j].position - Particles[i].position;
    float r = length(rij);

    if (r <= rCut && r > 1e-6) {
        float s = pow(sigma / r, 6);
        float f = min(24 * epsilon * s / (r * r) * (2 * s - 1), 100000.0);
        // float f = 24 * epsilon * s / (r * r) * (2 * s - 1);
        Particles[i].force -= f * rij;
    }
}

void ComputeF(uint i) {
    Particles[i].force = 0;

    uint2 cell = uint2(Particles[i].cell);

    int distance = 1;

    for (int dx = -distance; dx <= distance; dx++) {
        for (int dy = -distance; dy <= distance; dy++) {
            int2 neighbourCell = cell + int2(dx, dy);
            if (neighbourCell.x < 0 || neighbourCell.x >= gridSize.x || neighbourCell.y < 0 || neighbourCell.y >= gridSize.y) {
                continue;
            }

            uint neighbourCellIndex = GetCellIndex(neighbourCell);

            uint startIndex = CellOffsets[neighbourCellIndex];
            uint endIndex = CellOffsets[neighbourCellIndex + 1];

            for (uint j = startIndex; j < endIndex; j++) {
                uint particleID = SortedParticleIndices[j];
                if (i != particleID) Force(i, particleID);
            }
        }
    }
}

void AddGravity(uint i) {
    if (Particles[i].id == 1 & SimulationTime[0].totalTime < 15.0) return;
    Particles[i].force += Particles[i].mass * g;
}

void ComputeBounds(uint i) {
    float2 dMin = Particles[i].position - bounds.xz;
    float2 dMax = bounds.yw - Particles[i].position;

    float minRestrain = 0.01 * sigma;

    // --- Bottom / Left Wall Interaction ---
    float2 rMin = max(dMin, minRestrain);
    float2 sMin = pow(0.5 * sigma / rMin, 6);
    float2 fMin = min(24.0 * epsilon * 0.5 / rMin * sMin * (2.0 * sMin - 1.0), 100000.0.xx);

    Particles[i].force += max(fMin, 0.0); 

    // --- Top / Right Wall Interaction ---
    float2 rMax = max(dMax, minRestrain);
    float2 sMax = pow(0.5 * sigma / rMax, 6);
    float2 fMax = min(24.0 * epsilon * 0.5 / rMax * sMax * (2.0 * sMax - 1.0), 100000.0.xx);

    Particles[i].force -= max(fMax, 0.0);
}

[numthreads(256, 1, 1)] 
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint Index = dispatchThreadID.x;
    if (Index >= (uint)n) return;

    ComputeF(Index);
    ComputeBounds(Index);
    AddGravity(Index);

    float dt = SimulationTime[0].dt;
    float2 a = Particles[Index].force / Particles[Index].mass;
    Particles[Index].velocity += 0.5 * dt * a;
}