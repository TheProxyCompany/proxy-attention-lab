// primitive_factory.cpp
// Implementation of the primitive factory pattern
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include "pal_core/primitive_factory.hpp"
#include <stdexcept>

namespace pal::cpp {

std::unordered_map<std::string, PrimitiveFactory::PrimitiveCreator>&
PrimitiveFactory::get_registry() {
    static std::unordered_map<std::string, PrimitiveCreator> registry;
    return registry;
}

void PrimitiveFactory::register_primitive(const std::string& name, PrimitiveCreator creator) {
    get_registry()[name] = creator;
}

std::unique_ptr<mx::Primitive> PrimitiveFactory::create(
    const std::string& name,
    const PrimitiveConfig& config
) {
    auto& registry = get_registry();
    auto it = registry.find(name);
    if (it == registry.end()) {
        throw std::runtime_error("Primitive type not registered: " + name);
    }
    return it->second(config);
}

bool PrimitiveFactory::is_registered(const std::string& name) {
    return get_registry().find(name) != get_registry().end();
}

std::vector<std::string> PrimitiveFactory::get_registered_types() {
    std::vector<std::string> types;
    for (const auto& [name, _] : get_registry()) {
        types.push_back(name);
    }
    return types;
}

} // namespace pal::cpp
