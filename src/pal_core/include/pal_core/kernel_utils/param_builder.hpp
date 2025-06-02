#pragma once
// param_builder.hpp
// Base class for kernel parameter builders
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <Metal/Metal.hpp>
#include <cstddef>

namespace pal::cpp {

// Base class for building kernel-specific parameters
template<typename ParamsT>
class KernelParamBuilder {
public:
    virtual ~KernelParamBuilder() = default;

    // Build parameters from device and constraints
    virtual ParamsT build(MTL::Device* device) = 0;

protected:
    // Align a value to the specified alignment
    static size_t align_to(size_t value, size_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }
};

} // namespace pal::cpp
