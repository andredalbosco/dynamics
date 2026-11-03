#pragma once

#include <Eigen/Core>
#include <vector>

namespace xpbd {

struct Particle {
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d prevPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    double invMass = 1;
};

struct DistanceConstraint {
    int i, j;
    double restLength;
    double compliance = 0;
    double lambda = 0;

    void project(std::vector<Particle>& particles, double h);
};

struct Solver {
    std::vector<Particle> particles;
    std::vector<DistanceConstraint> constraints;

    Eigen::Vector3d gravity {0, -9.81, 0};
    int substeps = 10;
    double dt = 1.0 / 60;

    void predictPositions(double h);

    void updateVelocities(double h);

    void step();
};

} // namespace xpbd
