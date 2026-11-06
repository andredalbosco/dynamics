#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace xpbd {

using Vector2 = Eigen::Vector2d;
using ParticleIndex = std::size_t;
using ConstraintIndex = std::size_t;

struct Particle {
    Vector2 position = Vector2::Zero();
    Vector2 previousPosition = Vector2::Zero();
    Vector2 velocity = Vector2::Zero();
    double inverseMass = 0.0;

    bool isFixed() const { return inverseMass == 0.0; }
};

struct DistanceConstraint {
    ParticleIndex first = 0;
    ParticleIndex second = 0;
    double restLength = 0.0;
    double compliance = 0.0;
    double lambda = 0.0;

    double value(const std::vector<Particle>& particles) const;
    void project(std::vector<Particle>& particles, double timeStep);
};

class Solver {
public:
    ParticleIndex addParticle(const Vector2& position, double mass);
    ConstraintIndex addDistanceConstraint(
        ParticleIndex first,
        ParticleIndex second,
        double compliance = 0.0
    );
    ConstraintIndex addDistanceConstraint(
        ParticleIndex first,
        ParticleIndex second,
        double restLength,
        double compliance
    );

    void clear();
    void step(
        double timeStep,
        int substeps,
        int iterations,
        const Vector2& gravity
    );

    std::vector<Particle>& particles() { return particles_; }
    const std::vector<Particle>& particles() const { return particles_; }
    std::vector<DistanceConstraint>& constraints() { return constraints_; }
    const std::vector<DistanceConstraint>& constraints() const { return constraints_; }

    double maximumConstraintError() const;

private:
    void validateParticleIndex(ParticleIndex index) const;

    std::vector<Particle> particles_;
    std::vector<DistanceConstraint> constraints_;
};

} // namespace xpbd
