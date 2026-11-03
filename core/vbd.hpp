#pragma once

#include <Eigen/Core>
#include <vector>

namespace vbd {

struct Particle {
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d prevPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    double invMass = 1;
    Eigen::Vector3d inertialPos = Eigen::Vector3d::Zero();
};

struct SpringEnergy {
    int i, j;
    double restLength;
    double stiffness;

    int neighbor(int vertex) const;
};

double springEnergyValue(
    const Eigen::Vector3d& delta,
    double restLength,
    double stiffness
);

void springEnergyDerivatives(
    const Eigen::Vector3d& delta,
    double restLength,
    double stiffness,
    Eigen::Vector3d& gradient,
    Eigen::Matrix3d& hessian
);

struct Solver {
    std::vector<Particle> particles;
    std::vector<SpringEnergy> energies;

    Eigen::Vector3d gravity {0, -9.81, 0};
    int substeps = 10;
    double dt = 1.0 / 60;

    void predictPositions(double h);

    // Add contributions to gradient and hessian.
    void inertialDerivatives(
        int i,
        double inertia,
        Eigen::Vector3d& gradient,
        Eigen::Matrix3d& hessian
    ) const;

    void energyDerivatives(
        int i,
        Eigen::Vector3d& gradient,
        Eigen::Matrix3d& hessian
    ) const;

    double objectiveValue(
        int i,
        const Eigen::Vector3d& position,
        double inertia
    ) const;

    // Initialize outputs and assemble the objective derivatives.
    void objectiveDerivatives(
        int i,
        double inertia,
        Eigen::Vector3d& gradient,
        Eigen::Matrix3d& hessian
    ) const;

    Eigen::Vector3d lineSearch(int i, Eigen::Vector3d dx, double inertia) const;

    void optimizeVertex(int i, double h);

    void updateVelocities(double h);

    void step();
};

} // namespace vbd
