#include <metal_stdlib>
#include "paged_attention.h"

using namespace metal;

// This file primarily serves to include the header where the kernel is defined.
// If paged_attn_kernel were templated, you would add explicit instantiations here, like:
// template [[host_name("paged_attn_kernel_float")]] // Example if templated
//   void paged_attn_kernel<float>(...);             // Example if templated

// Since our current kernel isn't templated, simply including the header
// makes the [[kernel]] function available for compilation into the .metallib.
