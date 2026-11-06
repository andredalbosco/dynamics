#pragma once

#include <Eigen/Core>
#include <vector>

// 2D elastic MLS-MPM/APIC, following Hu et al. (2018) and their mls-mpm88 example.
// Quadratic B-splines, fixed-corotated elasticity, frictionless grid walls; no plasticity.

using MpmVec = Eigen::Vector2d;
using MpmMat = Eigen::Matrix2d;

struct MpmParticle {
    MpmVec pos, vel = MpmVec::Zero();
    MpmMat affine = MpmMat::Zero(), deformation = MpmMat::Identity();
    double mass, volume;
    int group = 0;
};
struct MpmNode {
    double mass = 0;
    MpmVec momentum = MpmVec::Zero(), velocity = MpmVec::Zero();
};

struct MpmSimulation {
    static constexpr int resolution = 64;
    static constexpr double dx = 1.0 / resolution;
    double young = 400, poisson = 0.2, density = 1;
    double maxSubstep = 2e-4, time = 0;
    MpmVec gravity {0, -9.81};
    std::vector<MpmParticle> particles;
    std::vector<MpmNode> grid;
    int lastSubsteps = 0;

    MpmSimulation();

    double mu() const;
    double lambda() const;

    MpmMat stress(const MpmMat& F) const;

    struct Stencil {
        Eigen::Vector2i base;
        MpmVec fraction, weight[3];
    };
    static Stencil stencil(const MpmVec& pos);
    MpmNode& node(int i, int j);

    void particleToGrid(double h);

    void updateGrid(double h, bool walls = true);

    void gridToParticle(double h);

    void step(double dt);
};
