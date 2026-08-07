struct Particle {
    float mass;
    float2 position;
    float2 velocity;
    float2 force;
    uint id;
};

struct TimeData {
    uint frame;
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

RWStructuredBuffer<float> Energy : register(u1);

StructuredBuffer<uint> CellOffsets : register(t0);
StructuredBuffer<uint> SortedParticleIndices : register(t1);
StructuredBuffer<TimeData> Time : register(t2);

float map(float input, float inputStart, float inputEnd, float outputStart, float outputEnd) {
    float slope = 1.0 * (outputEnd - outputStart) / (inputEnd - inputStart);
    return outputStart + slope * (input - inputStart);
}

float2 map(float2 input, float2 inputStart, float2 inputEnd, float2 outputStart, float2 outputEnd) {
    return float2(
        map(input.x, inputStart.x, inputEnd.x, outputStart.x, outputEnd.x), 
        map(input.y, inputStart.y, inputEnd.y, outputStart.y, outputEnd.y));
}

void ComputeV(uint i) {
    double a = Time[0].dt * .5 / Particles[i].mass;
    Particles[i].velocity += a * Particles[i].force;
}

int2 GetCell(float2 position) {
    float2 normalizedPos = map(position, bounds.xz, bounds.yw, 0.xx, float2(gridSize.xy));
    return clamp((int2)normalizedPos, int2(0.xx), int2(gridSize.xy - 1));
}

uint GetCellIndex(uint2 cell) {
    return cell.x + cell.y * gridSize.x;
}

void Force(uint i, uint j) {
    float2 rij = Particles[j].position - Particles[i].position;
    float r2 = dot(rij, rij);
    float rCut2 = rCut * rCut;

    if (r2 <= rCut2 && r2 > 1e-12) {
        float invR2 = 1.0 / r2;
        float s2 = (sigma * sigma) * invR2;
        float s6 = s2 * s2 * s2;
        float f = 24.0 * epsilon * s6 * invR2 * (1.0 - 2.0 * s6);
        Particles[i].force += f * rij;
    }
}

void ComputeF(uint i) {
    int2 cell = GetCell(Particles[i].position);
    int distance = 1;

    int minX = max(0, cell.x - distance);
    int maxX = min((int)gridSize.x - 1, cell.x + distance);
    int minY = max(0, cell.y - distance);
    int maxY = min((int)gridSize.y - 1, cell.y + distance);

    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            uint neighbourCellIndex = GetCellIndex(uint2(x, y));

            uint startIndex = CellOffsets[neighbourCellIndex];
            uint endIndex = CellOffsets[neighbourCellIndex + 1];

            for (uint j = startIndex; j < endIndex; j++) {
                uint particleID = SortedParticleIndices[j];

                if (i != particleID) {
                    Force(i, particleID);
                }
            }
        }
    }
}

void ComputeBounds(uint i) {
    float2 dMin = Particles[i].position - bounds.xz;
    float2 dMax = bounds.yw - Particles[i].position;

    float minRestrain = 0.01 * sigma;

    // --- Bottom / Left Wall Interaction ---
    float2 rMin = max(dMin, minRestrain);
    float2 sMin = pow(0.5 * sigma / rMin, 6);
    float2 fMin = min(24.0 * epsilon * 0.5 / rMin * sMin * (2.0 * sMin - 1.0), 10000.0.xx);

    Particles[i].force += max(fMin, 0.0); 

    // --- Top / Right Wall Interaction ---
    float2 rMax = max(dMax, minRestrain);
    float2 sMax = pow(0.5 * sigma / rMax, 6);
    float2 fMax = min(24.0 * epsilon * 0.5 / rMax * sMax * (2.0 * sMax - 1.0), 10000.0.xx);

    Particles[i].force -= max(fMax, 0.0);
}

void AddGravity(uint i) {
    if (Particles[i].id == 1 && Time[0].frame * Time[0].dt < 15.0) return;
    Particles[i].force += Particles[i].mass * g.xy;
}

void CalculateEnergy(uint i) {
    double kineticEnergy = 0.5 * Particles[i].mass * dot(Particles[i].velocity, Particles[i].velocity);

    Energy[0] += kineticEnergy;
}

[numthreads(256, 1, 1)] 
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint Index = dispatchThreadID.x;

    if (Index >= (uint)n) return;

    Particles[Index].force = 0.xx;

    ComputeF(Index);
    ComputeBounds(Index);
    AddGravity(Index);

    CalculateEnergy(Index);

    ComputeV(Index);
}