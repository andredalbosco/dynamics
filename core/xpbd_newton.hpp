#pragma once

#include "pbd.hpp"

// XPBD (2016), Eqs. 8-14: Newton on BOTH position and constitutive residuals.
// Finite compliance allows exact elimination/recovery of multiplier increments.

struct XpbdNewton : ParticleSolver {
    using ParticleSolver::ParticleSolver;
    double tolerance = 1e-9;
    double primalResidual = 0, constitutiveResidual = 0;
    int lastIterations = 0;
    bool converged = true;

    struct Residual {
        std::vector<Eigen::Vector3d> primal;
        std::vector<double> constitutive;
        double merit = 0, primalMax = 0, constitutiveMax = 0;
    };

    Residual residual(double h) const;

    void solve(double h);

    void step(double dt);
};
