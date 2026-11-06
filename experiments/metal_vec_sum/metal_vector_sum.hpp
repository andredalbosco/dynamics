#pragma once

#include <cstddef>
#include <string>

bool metal_vector_sum(
    const float* a,
    const float* b,
    float* result,
    std::size_t count,
    std::string& error);
