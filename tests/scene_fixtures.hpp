#pragma once

#include "core/xpbd.hpp"
#include "core/mpm.hpp"

xpbd::Solver makeSpring();
xpbd::Solver makePendulum();
xpbd::Solver makeDoublePendulum();
struct ClothFixture {
    xpbd::Solver simulation;
    std::vector<Eigen::Vector3i> triangles;
    size_t stretchCount = 0;
};
ClothFixture makeClothFixture(int rows, int cols, double spacing, double particleMass);
xpbd::Solver makeCloth(int rows, int cols, double spacing, double particleMass);
MpmSimulation makeMpmScene();

// Build the scene before a solver applies defaults or allocates GPU buffers.
template<class Solver>
Solver makeCloth(int rows, int cols, double spacing, double particleMass) {
    return Solver {makeCloth(rows, cols, spacing, particleMass)};
}

// Preserve the CPU reference methods' separate stretch/bend material settings.
template<class Solver>
Solver makeReferenceCloth(int rows, int cols, double spacing, double particleMass) {
    auto fixture = makeClothFixture(rows, cols, spacing, particleMass);
    auto& constraints = fixture.simulation.constraints;
    std::vector<xpbd::DistanceConstraint> bending(constraints.begin() + fixture.stretchCount, constraints.end());
    constraints.resize(fixture.stretchCount);
    Solver solver {std::move(fixture.simulation)};
    solver.bendConstraints = std::move(bending);
    solver.rebuildConnectivity();
    return solver;
}
