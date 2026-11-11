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

groupshared float EnergyScan[256];

int2 GetCell(float2 position) {
    float2 normalizedPos = (position - bounds.xz) * gridSize / (bounds.yw - bounds.xz);
    return clamp((int2)normalizedPos, int2(0, 0), int2(gridSize.xy - 1));
}

uint GetCellIndex(uint2 cell) {
    return cell.x + cell.y * gridSize.x;
}

void Force(inout Particle self, Particle other) {
    float sigma = 0.5 * (self.sigma + other.sigma);
    float epsilon = sqrt(self.epsilon * other.epsilon);
    float2 rij = other.position - self.position;

    float width = bounds.y - bounds.x;
    if (rij.x >  0.5 * width) rij.x -= width;
    if (rij.x < -0.5 * width) rij.x += width;

    float r2 = dot(rij, rij);
    float rcut = 2.5 * sigma;

    if (r2 <= rcut * rcut && r2 > 1e-12) {
        float invR2 = 1.0 / r2;
        float s2 = (sigma * sigma) * invR2;
        float s6 = s2 * s2 * s2;
        float f = 24.0 * epsilon * s6 * invR2 * (1.0 - 2.0 * s6);

        self.force += f * rij;
    }
}

int WrapCell(int cell, int count) {
    cell %= count;
    if (cell < 0) cell += count;
    return cell;
}

void ComputeF(inout Particle self, uint i) {
    int2 cell = GetCell(self.position);
    int distance = 1;

    for (int x = cell.x - distance; x <= cell.x + distance; x++) {
        for (int y = cell.y - distance; y <= cell.y + distance; y++) {
            int wrappedX = WrapCell(x, (int)gridSize.x);

            if (y < 0 || y >= (int)gridSize.y)
            continue;

            uint neighbourCellIndex = GetCellIndex(uint2(wrappedX, y));

            uint startIndex = CellOffsets[neighbourCellIndex];
            uint endIndex = CellOffsets[neighbourCellIndex + 1];

            for (uint j = startIndex; j < endIndex; j++) {
                uint particleID = SortedParticleIndices[j];

                if (i != particleID) {
                    Force(self, Particles[particleID]);
                }
            }
        }
    }
}

void ComputeBounds(inout Particle self) {
    float bottomDistance = self.position.y - bounds.z;
    float topDistance = bounds.w - self.position.y;

    float minRestrain = 0.01 * self.sigma;

    float rBottom = max(bottomDistance, minRestrain);
    float sBottom = pow(0.5 * self.sigma / rBottom, 6);
    float fBottom = 24.0 * self.epsilon * 0.5 / rBottom * sBottom * (2.0 * sBottom - 1.0);

    float rTop = max(topDistance, minRestrain);
    float sTop = pow(0.5 * self.sigma / rTop, 6);
    float fTop = 24.0 * self.epsilon * 0.5 / rTop * sTop * (2.0 * sTop - 1.0);

    self.force.y += max(fBottom, 0.0);
    self.force.y -= max(fTop, 0.0);
}

void AddGravity(inout Particle self) {
    // if (Particles[i].id == 1 && Time[0].frame * Time[0].dt < 15.0) return;
    self.force += self.mass * gravity.xy;
    // Particles[i].force += Particles[i].mass * float2(0.0, -12.0);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint Index = dispatchThreadID.x;
    uint lane = groupThreadID.x;
    float energy = 0.0;

    if (Index < (uint)n) {
        Particle self = Particles[Index];
        self.force = float2(0.0, 0.0);

        ComputeF(self, Index);
        ComputeBounds(self);
        AddGravity(self);

        energy = 0.5 * self.mass * dot(self.velocity, self.velocity);
        float a = 0.5 * Time[0].dt / self.mass;
        self.velocity += a * self.force;

        Particles[Index] = self;
    }

    EnergyScan[lane] = energy;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) EnergyScan[lane] += EnergyScan[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0) Energy[groupID.x] = EnergyScan[0];
}
