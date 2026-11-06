#pragma once

#include <cuda_runtime.h>

void cudaCheck(cudaError_t error, const char* operation);

#define CUDA_CHECK(call) cudaCheck((call), #call)
