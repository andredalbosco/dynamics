#include "cuda_utils.hpp"

#include <stdexcept>
#include <string>

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
