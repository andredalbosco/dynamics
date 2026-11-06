#pragma once

#include "xpbd.hpp"
#include "vbd.hpp"

struct ParticleSolver {
    std::vector<vbd::Particle> particles;
    std::vector<xpbd::DistanceConstraint> stretchConstraints;
    std::vector<xpbd::DistanceConstraint> bendConstraints;

    Eigen::Vector3d gravity {0, -9.81, 0};
    int substeps = 30;

    int iterations = 10;
    std::vector<Eigen::Vector3d> targets;
    std::vector<std::vector<int>> incident;

    explicit ParticleSolver(xpbd::Solver state);

    int constraintCount() const;

    xpbd::DistanceConstraint& constraint(int e);

    const xpbd::DistanceConstraint& constraint(int e) const;

    // Call after changing topology; the interactive scene changes it only on reset.
    void rebuildConnectivity();

    double substepSize(double dt) const;

    void predict(double h);

    void updateVelocities(double h);

    static double stiffness(const xpbd::DistanceConstraint& c);

    double localEnergy(int i, const Eigen::Vector3d& x, double h) const;
};

// Muller et al. (2007): sequential mass-weighted distance projections.

struct Pbd : ParticleSolver {
    using ParticleSolver::ParticleSolver;
    double stretchStiffness = 1;
    double bendingStiffness = 0.01;

    static void project(vbd::Particle& a, vbd::Particle& b, double rest, double stiffness);

    void step(double dt);
};
