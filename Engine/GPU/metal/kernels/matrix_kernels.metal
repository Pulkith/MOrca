// kernels.metal
#include <metal_stdlib>
using namespace metal;

// vector addition kernel: result = inA + inB
kernel void vector_add(const device float *inA [[buffer(0)]],
                       const device float *inB [[buffer(1)]],
                       device float *result [[buffer(2)]], // Output buffer
                       uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] + inB[index];
}

// Vector scaling kernel: result = inVec * scaleFactor
// Note: scaleFactor passed as a single-element buffer (buffer(2))
kernel void vector_scale(const device float *inVec [[buffer(0)]],
                         device float *resultVec [[buffer(1)]], // Output buffer
                         const device float *scaleFactor [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    // Read the single float value from the scaleFactor buffer
    resultVec[index] = inVec[index] * scaleFactor[0];
}

// Long running benchmark kernel. DO NOT USE IN PRODUCTION
kernel void long_running_task(const device float *inA [[buffer(0)]],
                             const device float *inB [[buffer(1)]],
                             device float *result [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
   float temp = inA[index] + inB[index];
   for (int i = 0; i < 500; ++i) {
        temp = cos(temp) * sin(temp) + temp * 0.99;
   }
   result[index] = temp; 
}


// ------------------------------
// Vector Subtraction Kernel
// Computes: result = inA - inB
// ------------------------------
kernel void vector_subtract(const device float *inA [[buffer(0)]],
                              const device float *inB [[buffer(1)]],
                              device float *result [[buffer(2)]],
                              uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] - inB[index];
}

// ------------------------------
// Vector Multiplication Kernel
// Computes element-wise multiplication: result = inA * inB
// ------------------------------
kernel void vector_multiply(const device float *inA [[buffer(0)]],
                              const device float *inB [[buffer(1)]],
                              device float *result [[buffer(2)]],
                              uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * inB[index];
}

// ------------------------------
// Vector Division Kernel
// Computes element-wise division: result = inA / inB
// Note: Division by zero is not explicitly handled.
// ------------------------------
kernel void vector_divide(const device float *inA [[buffer(0)]],
                            const device float *inB [[buffer(1)]],
                            device float *result [[buffer(2)]],
                            uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] / inB[index];
}

// ------------------------------
// Vector Exponentiation Kernel
// Computes the exponential of each element: result = exp(inVec)
// ------------------------------
kernel void vector_exp(const device float *inVec [[buffer(0)]],
                       device float *resultVec [[buffer(1)]],
                       uint index [[thread_position_in_grid]])
{
    resultVec[index] = exp(inVec[index]);
}

// ------------------------------
// Vector Logarithm Kernel
// Computes the natural logarithm of each element: result = log(inVec)
// Note: The input values should be greater than 0.
// ------------------------------
kernel void vector_log(const device float *inVec [[buffer(0)]],
                       device float *resultVec [[buffer(1)]],
                       uint index [[thread_position_in_grid]])
{
    resultVec[index] = log(inVec[index]);
}

// ------------------------------
// Vector Square Root Kernel
// Computes the square root of each element: result = sqrt(inVec)
// Note: The input values should be non-negative.
// ------------------------------
kernel void vector_sqrt(const device float *inVec [[buffer(0)]],
                        device float *resultVec [[buffer(1)]],
                        uint index [[thread_position_in_grid]])
{
    resultVec[index] = sqrt(inVec[index]);
}

// ------------------------------
// Vector Dot Product Kernel
// Computes the dot product of two vectors using a parallel reduction.
// This kernel assumes that the entire vector fits into a single threadgroup.
// The constant 'tgroup_size' (provided in buffer(3)) should be set to the
// number of threads in the threadgroup.
// The final dot product is written to result[0] by thread 0.
// ------------------------------
kernel void vector_dot_product(const device float *inA [[buffer(0)]],
                               const device float *inB [[buffer(1)]],
                               device float *result [[buffer(2)]],
                               constant uint &tgroup_size [[buffer(3)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint gid [[thread_position_in_grid]])
{
    // Declare a threadgroup array to store partial sums.
    // Assuming maximum threadgroup size of 256.
    threadgroup float localSums[256];
    
    // Each thread calculates the product of the corresponding elements.
    float product = inA[gid] * inB[gid];
    localSums[tid] = product;
    
    // Synchronize to ensure all threads have written their product.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform a parallel reduction within the threadgroup.
    for (uint stride = tgroup_size / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            localSums[tid] += localSums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // The first thread writes the result of the reduction.
    if (tid == 0)
    {
        result[0] = localSums[0];
    }
}