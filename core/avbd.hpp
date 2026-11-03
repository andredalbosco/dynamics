#pragma once

#include <Eigen/Core>
#include <vector>

namespace avbd {

struct Particle {
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d prevPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    double invMass = 1;
    Eigen::Vector3d inertialPos = Eigen::Vector3d::Zero();
};

struct DistanceConstraint {
    int i, j;
    double restLength;
    double compliance = 0;
    double penalty = 0;
    double lambda = 0;
    double initialResidual = 0;

    int neighbor(int vertex) const;

    double residual(double length, double alpha) const;

    void warmStart(const std::vector<Particle>& particles, double alpha, double gamma);

    void update(const std::vector<Particle>& particles, double alpha, double beta);
};

double augmentedEnergyValue(
    double r,
    double penalty,
    double lambda
);

void augmentedEnergyDerivatives(
    const Eigen::Vector3d& delta,
    double r,
    double penalty,
    double lambda,
    Eigen::Vector3d& gradient,
    Eigen::Matrix3d& hessian
);

struct Solver {
    std::vector<Particle> particles;
    std::vector<DistanceConstraint> constraints;

    Eigen::Vector3d gravity {0, -9.81, 0};
    int substeps = 10;
    double dt = 1.0 / 60;

    double alpha = 0.95;
    double gamma = 0.99;
    double beta = 1e5;

    void predictPositions(double h);

    void inertialDerivatives(
        int i,
        double inertia,
        Eigen::Vector3d& gradient,
        Eigen::Matrix3d& hessian
    ) const;

    void constraintDerivatives(
        int i,
        Eigen::Vector3d& gradient,
        Eigen::Matrix3d& hessian
    ) const;

    double objectiveValue(
        int i,
        const Eigen::Vector3d& position,
        double inertia
    ) const;

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

} // namespace avbd
