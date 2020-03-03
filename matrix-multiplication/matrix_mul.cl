#define TILE_SIZE 32
#define ELEMENTS_PER_THREAD 4

/// N, N, K must be divisible by tile size
kernel void matrix_mul(
        global const float* a, // A[M][K]
        global const float* b, // B[K][N]
        global float* c,       // C[M][N]
        uint N,
        uint K,
        uint M
) {
    local float a_tile[TILE_SIZE][TILE_SIZE];
    local float b_tile[TILE_SIZE][TILE_SIZE];

    uint const col_matr = get_global_id(0);
    uint const row_matr = get_global_id(1) * ELEMENTS_PER_THREAD;

    uint const col_tile = get_local_id(0);
    uint const row_tile = get_local_id(1) * ELEMENTS_PER_THREAD;

    float sums[ELEMENTS_PER_THREAD];
    for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        sums[i] = 0;
    }

    for (uint tile = 0; tile < K / TILE_SIZE; ++tile) {
        for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            a_tile[row_tile + i][col_tile] = a[(row_matr + i) * K + (tile * TILE_SIZE + col_tile)];
            b_tile[row_tile + i][col_tile] = b[(tile * TILE_SIZE + row_tile + i) * N + col_matr];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            for (uint j = 0; j < TILE_SIZE; ++j) {
                sums[i] += a_tile[row_tile + i][j] * b_tile[j][col_tile];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        c[(row_matr + i) * N + col_matr] = sums[i];
    }
}
