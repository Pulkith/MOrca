#include <metal_stdlib>
using namespace metal;

// --- Example Kernels ---

// Simple Vector Addition (as used in SimpleVectorAdd task)
kernel void add_vectors(device const float* vecA [[buffer(0)]],
                        device const float* vecB [[buffer(1)]],
                        device float* result [[buffer(2)]],
                        uint index [[thread_position_in_grid]])
{
    result[index] = vecA[index] + vecB[index];
}


// Example: Matrix Multiplication (Naive)
// Assumes matrices A, B, C are laid out linearly in buffers
// A is MxK, B is KxN, C is MxN
struct MatrixParams {
    uint widthA; // K
    uint widthB; // N
    // uint heightA; // M (Implicit from grid size)
    // uint heightB; // K (Implicit?)
};

kernel void matrix_multiply(device const float* matrixA [[buffer(0)]],
                            device const float* matrixB [[buffer(1)]],
                            device float* matrixC [[buffer(2)]],
                            constant MatrixParams& params [[buffer(3)]],
                            uint2 tid [[thread_position_in_grid]]) // tid.x = column (n), tid.y = row (m)
{
    uint M = params.widthA; // This seems wrong based on typical notation, let's assume layout
    uint K = params.widthA; // Width of A = Inner dim
    uint N = params.widthB; // Width of B = Outer dim of B/C

    uint row = tid.y; // Row index in C (0..M-1)
    uint col = tid.x; // Col index in C (0..N-1)

    if (row >= M /* height of A/C */ || col >= N /* width of B/C */) {
        // Bounds check if grid is larger than matrix C
        // Need height of A/C passed in params or calculated from grid
        // Assuming grid matches C dimensions M x N for now
        // return; // This needs correct M calculation
    }


    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        // matrixA[row][k] -> matrixA[row * K + k]
        // matrixB[k][col] -> matrixB[k * N + col]
        sum += matrixA[row * K + k] * matrixB[k * N + col];
    }

    // matrixC[row][col] -> matrixC[row * N + col]
    matrixC[row * N + col] = sum;
}


// Add more general-purpose or example kernels here...