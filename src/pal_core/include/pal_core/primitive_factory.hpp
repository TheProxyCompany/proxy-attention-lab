#pragma once
// primitive_factory.hpp
// Factory pattern for creating kernel primitives
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

namespace mx = mlx::core;

namespace pal::cpp {

// Base configuration for all primitives
struct PrimitiveConfig {
    mx::Stream stream;
    std::unordered_map<std::string, int> int_params;
    std::unordered_map<std::string, bool> bool_params;
    std::unordered_map<std::string, float> float_params;
};

// Factory for creating kernel primitives
class PrimitiveFactory {
public:
    using PrimitiveCreator = std::function<std::unique_ptr<mx::Primitive>(const PrimitiveConfig&)>;

    // Register a primitive type with the factory
    static void register_primitive(const std::string& name, PrimitiveCreator creator);

    // Create a primitive by name
    static std::unique_ptr<mx::Primitive> create(
        const std::string& name,
        const PrimitiveConfig& config
    );

    // Check if a primitive type is registered
    static bool is_registered(const std::string& name);

    // Get list of all registered primitive types
    static std::vector<std::string> get_registered_types();

private:
    static std::unordered_map<std::string, PrimitiveCreator>& get_registry();
};

// Helper macro for registering primitives
#define REGISTER_PRIMITIVE(name, type) \
    static bool _registered_##type = []() { \
        PrimitiveFactory::register_primitive(name, \
            [](const PrimitiveConfig& config) -> std::unique_ptr<mx::Primitive> { \
                return std::make_unique<type>(config); \
            }); \
        return true; \
    }();

} // namespace pal::cpp
