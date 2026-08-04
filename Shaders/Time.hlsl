struct TimeData {
    float totalTime;
    float dt;
};

RWStructuredBuffer<TimeData> SimulationTime : register(u0);

[numthreads(1, 1, 1)]
void main() {
    SimulationTime[0].totalTime += SimulationTime[0].dt;
}