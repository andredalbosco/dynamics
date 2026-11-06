#include "metal_vector_sum.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

int main() {
    constexpr std::size_t count = 1 << 20;

    std::vector<float> a(count);
    std::vector<float> b(count);
    std::vector<float> result(count);

    for (std::size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i) * 0.25f;
        b[i] = 3.0f - static_cast<float>(i) * 0.125f;
    }

    std::string error;
    if (!metal_vector_sum(a.data(), b.data(), result.data(), count, error)) {
        std::cerr << "Metal vector sum failed: " << error << '\n';
        return 1;
    }

    float max_error = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        max_error = std::max(max_error, std::abs(result[i] - (a[i] + b[i])));
    }

    std::cout << "summed " << count << " floats on Metal\n"
              << "first: " << result.front() << '\n'
              << "last:  " << result.back() << '\n'
              << "max error: " << max_error << '\n';

    return max_error == 0.0f ? 0 : 1;
}
