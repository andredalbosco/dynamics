#include "mpm.hpp"

#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

MpmSimulation::MpmSimulation() : grid((resolution + 1) * (resolution + 1)) {}

double MpmSimulation::mu() const { return young / (2 * (1 + poisson)); }

double MpmSimulation::lambda() const {
    return young * poisson / ((1 + poisson) * (1 - 2 * poisson));
}

MpmMat MpmSimulation::stress(const MpmMat& F) const {
    const double J = F.determinant();
    if (!(J > 0) || !F.allFinite()) {
        throw std::runtime_error("MPM element inverted; reduce timestep or stiffness");
    }
    // Proper 2D polar rotation for an orientation-preserving deformation.
    const double a = F(0, 0) + F(1, 1), b = F(1, 0) - F(0, 1);
    const double norm = std::hypot(a, b);
    if (!(norm > 1e-14)) {
        throw std::runtime_error("Degenerate MPM polar decomposition");
    }
    MpmMat R;
    R << a, -b, b, a;
    const MpmMat rotation = R / norm;
    // P F^T, multiplied by reference volume in the P2G transfer.
    return 2 * mu() * (F - rotation) * F.transpose()
        + lambda() * J * (J - 1) * MpmMat::Identity();
}

MpmSimulation::Stencil MpmSimulation::stencil(const MpmVec& pos) {
    Stencil s;
    s.base = (pos / dx - MpmVec::Constant(0.5)).array().floor().cast<int>();
    if ((s.base.array() < 0).any() || (s.base.array() + 2 > resolution).any())
        throw std::runtime_error("MPM particle left the grid stencil domain");
    s.fraction = pos / dx - s.base.cast<double>();
    s.weight[0] = 0.5 * (MpmVec::Constant(1.5) - s.fraction).array().square().matrix();
    s.weight[1] = MpmVec::Constant(0.75)
        - (s.fraction - MpmVec::Constant(1)).array().square().matrix();
    s.weight[2] = 0.5 * (s.fraction - MpmVec::Constant(0.5)).array().square().matrix();
    return s;
}

MpmNode& MpmSimulation::node(int i, int j) { return grid[i * (resolution + 1) + j]; }

void MpmSimulation::particleToGrid(double h) {
    std::fill(grid.begin(), grid.end(), MpmNode {});
    for (const auto& p : particles) {
        const Stencil s = stencil(p.pos);
        const MpmMat affine = p.mass * p.affine
            - h * p.volume * 4 / (dx * dx) * stress(p.deformation);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const double w = s.weight[i].x() * s.weight[j].y();
                const MpmVec offset = dx * (MpmVec(i, j) - s.fraction);
                auto& g = node(s.base.x() + i, s.base.y() + j);
                g.mass += w * p.mass;
                g.momentum += w * (p.mass * p.vel + affine * offset);
            }
        }
    }
}

void MpmSimulation::updateGrid(double h, bool walls) {
    for (int i = 0; i <= resolution; ++i) {
        for (int j = 0; j <= resolution; ++j) {
            auto& g = node(i, j);
            if (g.mass == 0)
                continue;
            g.velocity = g.momentum / g.mass + h * gravity;
            if (!walls)
                continue;
            if (i < 3)
                g.velocity.x() = std::max(0.0, g.velocity.x());
            if (i > resolution - 3)
                g.velocity.x() = std::min(0.0, g.velocity.x());
            if (j < 3)
                g.velocity.y() = std::max(0.0, g.velocity.y());
            if (j > resolution - 3)
                g.velocity.y() = std::min(0.0, g.velocity.y());
        }
    }
}

void MpmSimulation::gridToParticle(double h) {
    for (auto& p : particles) {
        const Stencil s = stencil(p.pos);
        p.vel.setZero();
        p.affine.setZero();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const double w = s.weight[i].x() * s.weight[j].y();
                const MpmVec offset = dx * (MpmVec(i, j) - s.fraction);
                const auto& g = node(s.base.x() + i, s.base.y() + j);
                p.vel += w * g.velocity;
                p.affine += 4 / (dx * dx) * w * g.velocity * offset.transpose();
            }
        }
        p.pos += h * p.vel;
        p.deformation = (MpmMat::Identity() + h * p.affine) * p.deformation;
        if (!p.pos.allFinite() || !p.vel.allFinite() || !p.affine.allFinite() ||
            !p.deformation.allFinite() || !(p.deformation.determinant() > 0))
            throw std::runtime_error("Invalid MPM state; reduce timestep or stiffness");
    }
}

void MpmSimulation::step(double dt) {
    if (!(dt > 0) || !std::isfinite(dt) || !(young > 0) || poisson < 0 || poisson >= 0.5)
        throw std::runtime_error("Invalid MPM timestep or material");
    double remaining = dt;
    lastSubsteps = 0;
    while (remaining > 1e-14) {
        double speed = 0, deformation = 1;
        for (const auto& p : particles) {
            speed = std::max(speed, p.vel.norm() + dx * p.affine.norm());
            deformation = std::max(deformation, p.deformation.norm());
        }
        const double wave = std::sqrt((lambda() + 2 * mu()) / density);
        // Conservative explicit-step heuristic, not an inversion-free guarantee.
        const double h = std::min(remaining,
            std::min(maxSubstep, 0.2 * dx / (speed + deformation * wave + 1e-12)));
        if (!(h > 1e-10) || ++lastSubsteps > 100000)
            throw std::runtime_error("MPM timestep collapsed");
        particleToGrid(h);
        updateGrid(h);
        gridToParticle(h);
        remaining -= h;
        time += h;
    }
}
