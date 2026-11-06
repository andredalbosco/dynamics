#pragma once

#include "parallel_cloth.hpp"
#include "utils/cuda_utils.hpp"

struct FallingDeviceParticle { float3 pos, previous, velocity; float invMass; };
struct CudaParallelCloth : ParallelCloth {
    FallingDeviceParticle* deviceParticles = nullptr;
    Link* deviceLinks = nullptr;
    float3* deviceCorrections = nullptr;
    float3* deviceNormals = nullptr;
    int* deviceOffsets = nullptr;
    int* deviceIncident = nullptr;
    int* deviceTriangleOffsets = nullptr;
    int* deviceTriangleIncident = nullptr;
    int3* deviceTriangles = nullptr;

    std::vector<FallingDeviceParticle> transfer;
    std::vector<float3> normalTransfer;

    explicit CudaParallelCloth(int cellsX = 140, int cellsY = 140,
        float spacing = 0.01f, float height = 2.2f);
    ~CudaParallelCloth();
    CudaParallelCloth(const CudaParallelCloth&) = delete;
    CudaParallelCloth& operator=(const CudaParallelCloth&) = delete;
    CudaParallelCloth(CudaParallelCloth&& other) noexcept;
    CudaParallelCloth& operator=(CudaParallelCloth&& other) noexcept;

    void uploadParticles();
    void step(float dt);
    void updateNormals();
};
