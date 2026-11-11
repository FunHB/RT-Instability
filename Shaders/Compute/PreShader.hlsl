struct TimeData {
    uint frame;
    float dt;
};

RWStructuredBuffer<TimeData> Time : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    Time[0].frame += 1;
}
