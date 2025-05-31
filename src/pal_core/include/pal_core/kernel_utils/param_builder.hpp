#pragma once
// param_builder.hpp
// Base class for kernel parameter builders
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <Metal/Metal.hpp>
#include <cstddef>

namespace pal::cpp {

// Memory constraints for kernel execution
struct MemoryConstraints {
    size_t total_threadgroup_memory;
    size_t fixed_memory_usage;
    size_t alignment_requirement;

    size_t available_memory() const {
        return (total_threadgroup_memory > fixed_memory_usage) ?
               (total_threadgroup_memory - fixed_memory_usage) : 0;
    }
};

// Base class for building kernel-specific parameters
template<typename ParamsT>
class KernelParamBuilder {
public:
    virtual ~KernelParamBuilder() = default;

    // Build parameters from device and constraints
    virtual ParamsT build(MTL::Device* device) = 0;

protected:
    // Helper to get memory constraints from device
    MemoryConstraints get_memory_constraints(MTL::Device* device, size_t fixed_usage) {
        MemoryConstraints constraints;
        constraints.total_threadgroup_memory = device->maxThreadgroupMemoryLength();
        constraints.fixed_memory_usage = fixed_usage;
        constraints.alignment_requirement = 64; // Common alignment for Metal
        return constraints;
    }

    // Align a value to the specified alignment
    static size_t align_to(size_t value, size_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }
};

} // namespace pal::cpp
