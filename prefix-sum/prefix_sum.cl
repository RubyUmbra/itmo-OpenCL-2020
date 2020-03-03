#define TILE_SIZE 4096

kernel void prefix_sum(uint N, global const float* gA, global float* gB) {
    uint j = get_local_id(0);
    local float localA[TILE_SIZE];

    localA[j] = gA[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint _2i = 1; _2i < N; _2i *= 2) {
        float l = localA[j];
        if (j >= _2i) {
            l += localA[j - _2i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        localA[j] = l;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    gB[j] = localA[j];
}
