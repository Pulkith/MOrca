// kernels.metal
#include <metal_stdlib>
using namespace metal;

// Simple vector addition kernel: result = inA + inB
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

// Kernel that takes longer (simulated) - just does an add
kernel void long_running_task(const device float *inA [[buffer(0)]],
                             const device float *inB [[buffer(1)]],
                             device float *result [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
   // Simulate work by doing the calculation many times - GPU is fast so may not take long!
   float temp = inA[index] + inB[index];
   /*
   // This loop might get optimized away or still be too fast.
   // True long tasks depend on computation complexity.
   for (int i = 0; i < 500; ++i) {
        temp = cos(temp) * sin(temp) + temp * 0.99;
   }
   */
   result[index] = temp; // Store final result
}

