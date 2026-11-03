#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <vector>

namespace pd {

struct Particles {
    Eigen::MatrixXd pos;
    Eigen::MatrixXd prevPos;
    Eigen::MatrixXd vel;
    Eigen::VectorXd invMass;
    Eigen::MatrixXd inertialPos;

    void resize(int count);
};

struct SpringEnergy {
    int i, j;
    double restLength;
    double stiffness;
};

// Add the spring contribution to gradient.
void springEnergyGradient(
    const Eigen::RowVector3d& delta,
    double restLength,
    double stiffness,
    const Eigen::RowVector3d& previousDelta,
    Eigen::RowVector3d& gradient
);

struct Cache {
    double factorizedH = 0;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
};

struct Solver {
    Particles particles;
    std::vector<SpringEnergy> energies;

    Eigen::Vector3d gravity {0, -9.81, 0};
    int substeps = 10;
    double dt = 1.0 / 60;

    Cache cache;

    Eigen::SparseMatrix<double> approximateHessian() const;

    // Rebuild after changing timestep, masses, pins, stiffness, or topology.
    void factorize();

    void predictPositions(double h);

    // Add contributions to gradient.
    void inertialGradient(Eigen::MatrixXd& gradient) const;

    void energyGradient(Eigen::MatrixXd& gradient) const;

    // Initialize gradient and assemble the objective gradient.
    void objectiveGradient(Eigen::MatrixXd& gradient) const;

    void updateVelocities(double h);

    void step();
};

} // namespace pd
