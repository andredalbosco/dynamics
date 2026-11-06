#include "xpbd.hpp"

#include <cmath>
#include <iostream>

namespace {

bool near(double actual, double expected, double tolerance)
{
    if (std::abs(actual - expected) <= tolerance)
    {
        return true;
    }

    std::cerr << "expected " << expected << ", got " << actual << '\n';
    return false;
}

bool hardDistanceProjection()
{
    xpbd::Solver solver;
    const auto fixed = solver.addParticle({0.0, 0.0}, 0.0);
    const auto moving = solver.addParticle({2.0, 0.0}, 1.0);
    solver.addDistanceConstraint(fixed, moving, 1.0, 0.0);

    solver.step(1.0 / 60.0, 1, 1, {0.0, 0.0});

    return near(solver.particles()[moving].position.x(), 1.0, 1.0e-12)
        && near(solver.maximumConstraintError(), 0.0, 1.0e-12);
}

bool pendulumLengthPreservation()
{
    xpbd::Solver solver;
    const auto fixed = solver.addParticle({0.0, 0.0}, 0.0);
    const auto moving = solver.addParticle({1.0, 0.0}, 1.0);
    solver.addDistanceConstraint(fixed, moving, 1.0, 0.0);

    for (int frame = 0; frame < 240; ++frame)
    {
        solver.step(1.0 / 60.0, 4, 4, {0.0, -9.81});
    }

    return near(solver.maximumConstraintError(), 0.0, 1.0e-10)
        && near(solver.particles()[fixed].position.norm(), 0.0, 1.0e-12);
}

} // namespace

int main()
{
    if (!hardDistanceProjection() || !pendulumLengthPreservation())
    {
        return 1;
    }

    return 0;
}
