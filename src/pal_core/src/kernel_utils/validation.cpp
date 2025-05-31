// validation.cpp
// Implementation of common validation utilities
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include "pal_core/kernel_utils/validation.hpp"
#include <stdexcept>
#include <sstream>
#include <spdlog/spdlog.h>

namespace pal::cpp::kernel_utils {

void ValidationUtils::check_dtype(const mx::array& arr, mx::Dtype expected,
                                const std::string& array_name) {
    if (arr.dtype() != expected) {
        std::ostringstream oss;
        oss << array_name << " must have expected dtype, but types don't match";
        throw std::invalid_argument(oss.str());
    }
}

void ValidationUtils::check_ndim(const mx::array& arr, int expected_ndim,
                               const std::string& array_name) {
    if (arr.ndim() != expected_ndim) {
        std::ostringstream oss;
        oss << array_name << " must be " << expected_ndim
            << "D, but got " << arr.ndim() << "D";
        throw std::invalid_argument(oss.str());
    }
}

void ValidationUtils::check_shape_at_dim(const mx::array& arr, int dim, int expected_size,
                                       const std::string& array_name) {
    if (dim >= arr.ndim() || dim < 0) {
        throw std::invalid_argument("Invalid dimension index for " + array_name);
    }
    if (arr.shape(dim) != expected_size) {
        std::ostringstream oss;
        oss << array_name << " dimension " << dim << " must be "
            << expected_size << ", but got " << arr.shape(dim);
        throw std::invalid_argument(oss.str());
    }
}

void ValidationUtils::check_divisibility(int value, int divisor,
                                       const std::string& value_name) {
    if (value % divisor != 0) {
        std::ostringstream oss;
        oss << value_name << " (" << value << ") must be divisible by " << divisor;
        throw std::invalid_argument(oss.str());
    }
}

void ValidationUtils::check_valid_data(const mx::array& arr,
                                     const std::string& array_name) {
    if (!arr.data<void>()) {
        throw std::runtime_error(array_name + " has null data pointer");
    }
}

void ValidationUtils::check_all_valid_data(const std::vector<mx::array>& arrays,
                                         const std::vector<std::string>& names) {
    if (arrays.size() != names.size()) {
        throw std::invalid_argument("Arrays and names vectors must have same size");
    }

    for (size_t i = 0; i < arrays.size(); ++i) {
        check_valid_data(arrays[i], names[i]);
    }
}

std::string KernelValidator::format_error(const std::string& kernel_name,
                                        const std::string& message) const {
    return "[" + kernel_name + " Validate] " + message;
}

} // namespace pal::cpp::kernel_utils
