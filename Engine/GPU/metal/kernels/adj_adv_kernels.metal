#include <metal_stdlib>
using namespace metal;

// ------------------------------
// Matrix Addition Kernel
// Computes element-wise addition: matrixC = matrixA + matrixB
// The matrices are flattened and the grid should be dispatched over the total number of elements.
// ------------------------------
kernel void matrix_add(const device float* matrixA [[buffer(0)]],
                         const device float* matrixB [[buffer(1)]],
                         device float* matrixC [[buffer(2)]],
                         uint index [[thread_position_in_grid]])
{
    matrixC[index] = matrixA[index] + matrixB[index];
}

// ------------------------------
// Matrix Multiplication Kernel
// Computes the product of two matrices:
// matrixC = matrixA * matrixB, where matrixA is of size (heightA x widthA)
// and matrixB is of size (widthA x widthB). The result matrixC has dimensions (heightA x widthB).
// The grid should be dispatched in a 2D configuration (using uint2 for thread positions).
// ------------------------------
kernel void matrix_multiply(const device float* matrixA [[buffer(0)]],
                              const device float* matrixB [[buffer(1)]],
                              device float* matrixC [[buffer(2)]],
                              constant uint &widthA [[buffer(3)]],
                              constant uint &heightA [[buffer(4)]],
                              constant uint &widthB [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if(gid.x < widthB && gid.y < heightA)
    {
        float sum = 0.0;
        // Iterate over the shared dimension (widthA)
        for(uint k = 0; k < widthA; k++)
        {
            sum += matrixA[gid.y * widthA + k] * matrixB[k * widthB + gid.x];
        }
        matrixC[gid.y * widthB + gid.x] = sum;
    }
}

// ------------------------------
// Matrix Transpose Kernel
// Transposes a given matrix:
// outMatrix = transpose(inMatrix)
// 'width' and 'height' are the dimensions of the input matrix.
// The grid should be dispatched over the dimensions of the input matrix.
// ------------------------------
kernel void matrix_transpose(const device float* inMatrix [[buffer(0)]],
                               device float* outMatrix [[buffer(1)]],
                               constant uint &width [[buffer(2)]],
                               constant uint &height [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]])
{
    if(gid.x < width && gid.y < height)
    {
        // Transpose: the element at (y, x) in the input becomes (x, y) in the output.
        outMatrix[gid.x * height + gid.y] = inMatrix[gid.y * width + gid.x];
    }
}