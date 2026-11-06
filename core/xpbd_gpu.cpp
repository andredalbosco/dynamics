#include "xpbd_gpu.hpp"

#include <utility>
#include <cmath>
#include <stdexcept>
#include <string>

namespace xpbd {

ConstraintColors::ConstraintColors(const std::vector<DistanceConstraint>& constraints,
    size_t particleCount) {
    std::vector<std::vector<int>> batches;
    std::vector<std::vector<bool>> used;
    for (int i = 0; i < int(constraints.size()); ++i) {
        const auto& c = constraints[i];
        size_t color = 0;
        while (color < batches.size() && (used[color][c.i] || used[color][c.j]))
            ++color;
        if (color == batches.size()) {
            batches.emplace_back();
            used.emplace_back(particleCount, false);
        }
        batches[color].push_back(i);
        used[color][c.i] = used[color][c.j] = true;
    }
    for (const auto& batch : batches) {
        order.insert(order.end(), batch.begin(), batch.end());
        offsets.push_back(int(order.size()));
    }
}

static __global__ void predict(DeviceParticle* particles, int count, double h, double ratio,
    double gx, double gy, double gz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    auto& p = particles[i];
    const double gravity[] = {gx, gy, gz};
    for (int k = 0; k < 3; ++k) {
        const double displacement = (p.x[k] - p.previous[k]) * ratio;
        p.previous[k] = p.x[k];
        if (p.invMass == 0) continue;
        p.x[k] += displacement + gravity[k] * h * h;
    }
}

static __global__ void project(DeviceParticle* particles, DistanceConstraint* constraints,
    int count, double h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    auto& c = constraints[i];
    c.lambda = 0; // One sweep, multipliers reset each substep.
    auto& a = particles[c.i];
    auto& b = particles[c.j];
    double delta[3];
    double squared = 0;
    for (int k = 0; k < 3; ++k) {
        delta[k] = b.x[k] - a.x[k];
        squared += delta[k] * delta[k];
    }
    const double length = sqrt(squared);
    const double weight = a.invMass + b.invMass;
    if (length < 1e-12 || weight == 0)
        return;
    c.lambda = -(length - c.restLength) / (weight + c.compliance / (h * h));
    for (int k = 0; k < 3; ++k) {
        const double gradient = delta[k] * (1 / length);
        a.x[k] -= gradient * (a.invMass * c.lambda);
        b.x[k] += gradient * (b.invMass * c.lambda);
    }
}

CudaSolver::CudaSolver(Solver state)
    : Solver(std::move(state)),
        colors(constraints, particles.size()),
        hostParticles(particles.size()), hostConstraints(constraints.size()) {
    try {
        if (!particles.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceParticles), (particles.size()) * sizeof(DeviceParticle)));
        if (!constraints.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceConstraints), (constraints.size()) * sizeof(DistanceConstraint)));
    } catch (...) {
        cudaFree(deviceParticles);
        cudaFree(deviceConstraints);
        throw;
    }
}

void CudaSolver::step(double dt) {
    if (!(dt > 0) || !std::isfinite(dt) || substeps < 1)
        throw std::invalid_argument("Positive finite dt and positive substeps required");
    // Host state is the viewer/edit interface. Upload once per frame, not per substep.
    // This includes grabs/releases and edited compliance; paused edits persist on host.
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& d = hostParticles[i];
        const auto& p = particles[i];
        for (int k = 0; k < 3; ++k) {
            d.x[k] = p.pos[k];
            d.previous[k] = p.prevPos[k];
        }
        d.invMass = p.invMass;
    }
    CUDA_CHECK(cudaMemcpy(deviceParticles, hostParticles.data(),
        hostParticles.size() * sizeof(DeviceParticle), cudaMemcpyHostToDevice));
    auto upload = [](const auto& source, auto& host, auto& device, const auto& colors) {
        for (size_t i = 0; i < source.size(); ++i)
            host[i] = source[colors.order[i]];
        if (!host.empty())
            CUDA_CHECK(cudaMemcpy(device, host.data(),
                host.size() * sizeof(DistanceConstraint), cudaMemcpyHostToDevice));
    };
    upload(constraints, hostConstraints, deviceConstraints, colors);
    const double h = dt / substeps;
    const int blocks = (int(particles.size()) + 127) / 128;
    auto solve = [&](auto& device, const auto& colors) {
        for (size_t color = 1; color < colors.offsets.size(); ++color) {
            const int first = colors.offsets[color - 1];
            const int count = colors.offsets[color] - first;
            project<<<(count + 127) / 128, 128>>>(deviceParticles,
                device + first, count, h);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    for (int substep = 0; substep < substeps; ++substep) {
        predict<<<blocks, 128>>>(deviceParticles, int(particles.size()), h,
            substep == 0 && previousH > 0 ? h / previousH : 1,
            gravity.x(), gravity.y(), gravity.z());
        CUDA_CHECK(cudaGetLastError());
        solve(deviceConstraints, colors);
    }
    // Same-stream launches order the batches. Check execution errors before host rendering.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hostParticles.data(), deviceParticles,
        hostParticles.size() * sizeof(DeviceParticle), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < particles.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            particles[i].pos[k] = hostParticles[i].x[k];
            particles[i].prevPos[k] = hostParticles[i].previous[k];
        }
    }
    auto download = [](auto& target, auto& host, auto& device, const auto& colors) {
        if (!host.empty())
            CUDA_CHECK(cudaMemcpy(host.data(), device,
                host.size() * sizeof(DistanceConstraint), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < host.size(); ++i)
            target[colors.order[i]].lambda = host[i].lambda;
    };
    download(constraints, hostConstraints, deviceConstraints, colors);
    previousH = h;
}

CudaSolver::~CudaSolver() {
    cudaFree(deviceParticles);
    cudaFree(deviceConstraints);
}

CudaSolver::CudaSolver(CudaSolver&& other) noexcept
    : Solver(std::move(other)),
      previousH(other.previousH),
      colors(std::move(other.colors)),
      deviceParticles(std::exchange(other.deviceParticles, nullptr)),
      deviceConstraints(std::exchange(other.deviceConstraints, nullptr)),
      hostParticles(std::move(other.hostParticles)),
      hostConstraints(std::move(other.hostConstraints)) {}

CudaSolver& CudaSolver::operator=(CudaSolver&& other) noexcept {
    if (this != &other) {
        std::swap(static_cast<Solver&>(*this), static_cast<Solver&>(other));
        std::swap(previousH, other.previousH);
        std::swap(colors, other.colors);
        std::swap(deviceParticles, other.deviceParticles);
        std::swap(deviceConstraints, other.deviceConstraints);
        std::swap(hostParticles, other.hostParticles);
        std::swap(hostConstraints, other.hostConstraints);
    }
    return *this;
}

} // namespace xpbd
