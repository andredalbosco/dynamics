#pragma once

#include "mpm.hpp"
#include "utils/cuda_utils.hpp"

struct MpmDeviceParticle { double x,y,vx,vy,C[4],F[4],mass,volume; };
struct MpmDeviceNode { double mass,mx,my,vx,vy; };
struct MpmDeviceStats { double speed,deformation; int invalid; };
struct CudaMpmSimulation : MpmSimulation {
    MpmDeviceParticle* deviceParticles = nullptr;
    MpmDeviceNode* deviceGrid = nullptr;
    MpmDeviceStats* deviceStats = nullptr;

    std::vector<MpmDeviceParticle> transfer;
    explicit CudaMpmSimulation(MpmSimulation state);
    ~CudaMpmSimulation();
    CudaMpmSimulation(const CudaMpmSimulation&) = delete;
    CudaMpmSimulation& operator=(const CudaMpmSimulation&) = delete;
    CudaMpmSimulation(CudaMpmSimulation&& other) noexcept;
    CudaMpmSimulation& operator=(CudaMpmSimulation&& other) noexcept;

    void step(double dt);
};
