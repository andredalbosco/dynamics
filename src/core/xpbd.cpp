#include "xpbd.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xpbd {

namespace {

constexpr double minimumLength = 1.0e-12;

} // namespace

double DistanceConstraint::value(const std::vector<Particle>& particles) const
{
    return (particles[first].position - particles[second].position).norm() - restLength;
}

void DistanceConstraint::project(std::vector<Particle>& particles, double timeStep)
{
    Particle& a = particles[first];
    Particle& b = particles[second];

    Vector2 direction = a.position - b.position;
    const double length = direction.norm();
    if (length <= minimumLength)
    {
        return;
    }
    direction /= length;

    const double scaledCompliance = compliance / (timeStep * timeStep);
    const double denominator = a.inverseMass + b.inverseMass + scaledCompliance;
    if (denominator <= 0.0)
    {
        return;
    }

    const double deltaLambda =
        -(length - restLength + scaledCompliance * lambda) / denominator;
    lambda += deltaLambda;

    a.position += a.inverseMass * deltaLambda * direction;
    b.position -= b.inverseMass * deltaLambda * direction;
}

ParticleIndex Solver::addParticle(const Vector2& position, double mass)
{
    if (mass < 0.0)
    {
        throw std::invalid_argument("particle mass cannot be negative");
    }

    Particle particle;
    particle.position = position;
    particle.previousPosition = position;
    particle.inverseMass = mass == 0.0 ? 0.0 : 1.0 / mass;
    particles_.push_back(particle);
    return particles_.size() - 1;
}

ConstraintIndex Solver::addDistanceConstraint(
    ParticleIndex first,
    ParticleIndex second,
    double compliance
)
{
    validateParticleIndex(first);
    validateParticleIndex(second);
    const double restLength =
        (particles_[first].position - particles_[second].position).norm();
    return addDistanceConstraint(first, second, restLength, compliance);
}

ConstraintIndex Solver::addDistanceConstraint(
    ParticleIndex first,
    ParticleIndex second,
    double restLength,
    double compliance
)
{
    validateParticleIndex(first);
    validateParticleIndex(second);
    if (first == second)
    {
        throw std::invalid_argument("distance constraint endpoints must differ");
    }
    if (restLength < 0.0)
    {
        throw std::invalid_argument("distance constraint rest length cannot be negative");
    }
    if (compliance < 0.0)
    {
        throw std::invalid_argument("distance constraint compliance cannot be negative");
    }

    constraints_.push_back({first, second, restLength, compliance, 0.0});
    return constraints_.size() - 1;
}

void Solver::clear()
{
    particles_.clear();
    constraints_.clear();
}

void Solver::step(
    double timeStep,
    int substeps,
    int iterations,
    const Vector2& gravity
)
{
    if (timeStep <= 0.0)
    {
        throw std::invalid_argument("time step must be positive");
    }
    if (substeps <= 0)
    {
        throw std::invalid_argument("substep count must be positive");
    }
    if (iterations <= 0)
    {
        throw std::invalid_argument("iteration count must be positive");
    }

    const double substepTime = timeStep / static_cast<double>(substeps);

    for (int substep = 0; substep < substeps; ++substep)
    {
        for (Particle& particle : particles_)
        {
            particle.previousPosition = particle.position;
            if (particle.isFixed())
            {
                particle.velocity.setZero();
                continue;
            }

            particle.velocity += substepTime * gravity;
            particle.position += substepTime * particle.velocity;
        }

        for (DistanceConstraint& constraint : constraints_)
        {
            constraint.lambda = 0.0;
        }

        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            for (DistanceConstraint& constraint : constraints_)
            {
                constraint.project(particles_, substepTime);
            }
        }

        for (Particle& particle : particles_)
        {
            if (!particle.isFixed())
            {
                particle.velocity =
                    (particle.position - particle.previousPosition) / substepTime;
            }
        }
    }
}

double Solver::maximumConstraintError() const
{
    double maximumError = 0.0;
    for (const DistanceConstraint& constraint : constraints_)
    {
        maximumError = std::max(maximumError, std::abs(constraint.value(particles_)));
    }
    return maximumError;
}

void Solver::validateParticleIndex(ParticleIndex index) const
{
    if (index >= particles_.size())
    {
        throw std::out_of_range("particle index is out of range");
    }
}

} // namespace xpbd
