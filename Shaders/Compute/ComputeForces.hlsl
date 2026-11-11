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
    float a = Time[0].dt * .5 / Particles[i].mass;
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
    float sigma = 0.5 * (Particles[i].sigma + Particles[j].sigma);
    float epsilon = sqrt(Particles[i].epsilon * Particles[j].epsilon);

    float2 rij = Particles[j].position - Particles[i].position;
    float r2 = dot(rij, rij);
    float rcut = 2.5 * sigma;

    if (r2 <= rcut * rcut && r2 > 1e-12) {
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

    float minRestrain = 0.01 * Particles[i].sigma;

    // --- Bottom / Left Wall Interaction ---
    float2 rMin = max(dMin, minRestrain);
    float2 qMin = 0.5 * Particles[i].sigma / rMin;
    float2 q2Min = qMin * qMin;
    float2 sMin = q2Min * q2Min * q2Min;
    float2 fMin = 24.0 * Particles[i].epsilon * 0.5 / rMin * sMin * (2.0 * sMin - 1.0);

    Particles[i].force += max(fMin, 0.0); 

    // --- Top / Right Wall Interaction ---
    float2 rMax = max(dMax, minRestrain);
    float2 qMax = 0.5 * Particles[i].sigma / rMax;
    float2 q2Max = qMax * qMax;
    float2 sMax = q2Max * q2Max * q2Max;
    float2 fMax = 24.0 * Particles[i].epsilon * 0.5 / rMax * sMax * (2.0 * sMax - 1.0);

    Particles[i].force -= max(fMax, float2(0.0, 0.0));
}

void AddGravity(uint i) {
    // if (Particles[i].id == 1 && Time[0].frame * Time[0].dt < 15.0) return;
    Particles[i].force += Particles[i].mass * gravity.xy;
    // Particles[i].force += Particles[i].mass * float2(0.0, -12.0);
}

void CalculateEnergy(uint i) {
    float kineticEnergy = 0.5 * Particles[i].mass * dot(Particles[i].velocity, Particles[i].velocity);

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