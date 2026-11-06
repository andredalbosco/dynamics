#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

namespace bw {

struct Particle {
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d prevPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    double invMass = 1;
};

struct SpringEnergy {
    int i, j;
    double restLength;
    double stiffness;
};

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

    // Add spring contributions to gradient and hessian.
    void energyDerivatives(Eigen::VectorXd& gradient, Eigen::SparseMatrix<double>& hessian) const;

    void solveVelocities(double h);

    void updatePositions(double h);

    void step();
};

} // namespace bw
