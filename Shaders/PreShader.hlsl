struct TimeData {
    uint frame;
    float dt;
};

RWStructuredBuffer<TimeData> Time : register(u0);
RWStructuredBuffer<float> Energy : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    Time[0].frame += 1;
    Energy[0] = 0;
}
