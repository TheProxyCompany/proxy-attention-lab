#pragma once
// validation.hpp
// Common validation utilities for kernel inputs
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <mlx/array.h>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace pal::cpp::kernel_utils {

// Common validation functions that can be used by multiple kernels
class ValidationUtils {
public:
    // Check if array has expected dtype
    static void check_dtype(const mx::array& arr, mx::Dtype expected,
                          const std::string& array_name);

    // Check if array has expected number of dimensions
    static void check_ndim(const mx::array& arr, int expected_ndim,
                         const std::string& array_name);

    // Check if array shape matches expected shape at specific dimensions
    static void check_shape_at_dim(const mx::array& arr, int dim, int expected_size,
                                 const std::string& array_name);

    // Check if a dimension is divisible by a value (common for vectorization requirements)
    static void check_divisibility(int value, int divisor,
                                 const std::string& value_name);

    // Validate array is not empty and has valid data pointer
    static void check_valid_data(const mx::array& arr,
                               const std::string& array_name);

    // Batch validation of multiple arrays having valid data pointers
    static void check_all_valid_data(const std::vector<mx::array>& arrays,
                                   const std::vector<std::string>& names);
};

// Base class for kernel-specific validators
class KernelValidator {
public:
    virtual ~KernelValidator() = default;

    // Override this to implement kernel-specific validation
    virtual void validate(const std::vector<mx::array>& inputs) = 0;

protected:
    // Utility to format error messages consistently
    std::string format_error(const std::string& kernel_name,
                           const std::string& message) const;
};

} // namespace pal::cpp::kernel_utils
