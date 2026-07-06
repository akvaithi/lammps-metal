#include <metal_stdlib>
using namespace metal;

kernel void kernel_zero(
    device int *mem [[buffer(0)]],
    constant int &numel [[buffer(1)]],
    uint ii [[thread_position_in_grid]]
) {
    if (ii < numel) mem[ii] = 0;
}

kernel void kernel_info(
    device int *info [[buffer(0)]]
) {
    info[0] = 900; // __CUDA_ARCH__
    info[1] = 0; // CONFIG_ID
    info[2] = 32; // SIMD_SIZE
    info[3] = 32; // MEM_THREADS
    info[4] = 0; // SHUFFLE_AVAIL
    info[5] = 0; // FAST_MATH
    info[6] = 1; // THREADS_PER_ATOM
    info[7] = 1; // THREADS_PER_CHARGE
    info[8] = 1; // THREADS_PER_THREE
    info[9] = 256; // BLOCK_PAIR
    info[10] = 256; // BLOCK_BIO_PAIR
    info[11] = 256; // BLOCK_ELLIPSE
    info[12] = 256; // PPPM_BLOCK_1D
    info[13] = 256; // BLOCK_NBOR_BUILD
    info[14] = 16; // BLOCK_CELL_2D (transpose tile; keep small, threadgroup tile is NxN+1)
    info[15] = 256; // BLOCK_CELL_ID
    info[16] = 8; // MAX_SHARED_TYPES
    info[17] = 8; // MAX_BIO_SHARED_TYPES
    info[18] = 0; // PPPM_MAX_SPLINE
    info[19] = 0; // NBOR_PREFETCH
}
