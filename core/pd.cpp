#include "pd.hpp"

namespace pd {

constexpr double epsilon = 1e-12;

void springEnergyGradient(
    const Eigen::RowVector3d& delta,
    double restLength,
    double stiffness,
    const Eigen::RowVector3d& previousDelta,
    Eigen::RowVector3d& gradient
) {
    Eigen::RowVector3d n = delta;
    if (n.norm() < epsilon) n = previousDelta;
    if (n.norm() < epsilon) n = Eigen::RowVector3d::UnitX();
    else n.normalize();

    gradient += stiffness * (delta - restLength * n);
}

void Particles::resize(int count) {
    pos = Eigen::MatrixXd::Zero(count, 3);
    prevPos = pos;
    vel = pos;
    inertialPos = pos;
    invMass = Eigen::VectorXd::Ones(count);
}

Eigen::SparseMatrix<double> Solver::approximateHessian() const {
    auto& ps = particles;
    std::vector<Eigen::Triplet<double>> entries;

    for (int i = 0; i < ps.pos.rows(); ++i) {
        const double inertia = ps.invMass[i] == 0 ? 1 : 1 / (ps.invMass[i] * cache.factorizedH * cache.factorizedH);
        entries.emplace_back(i, i, inertia);
    }

    for (const auto& e : energies) {
        if (ps.invMass[e.i] > 0) entries.emplace_back(e.i, e.i, e.stiffness);
        if (ps.invMass[e.j] > 0) entries.emplace_back(e.j, e.j, e.stiffness);
        if (ps.invMass[e.i] > 0 && ps.invMass[e.j] > 0) {
            entries.emplace_back(e.i, e.j, -e.stiffness);
            entries.emplace_back(e.j, e.i, -e.stiffness);
        }
    }

    Eigen::SparseMatrix<double> matrix(ps.pos.rows(), ps.pos.rows());
    matrix.setFromTriplets(entries.begin(), entries.end());
    return matrix;
}

void Solver::factorize() {
    cache.factorizedH = dt / substeps;
    cache.factor.compute(approximateHessian());
}

void Solver::predictPositions(double h) {
    auto& ps = particles;
    ps.prevPos = ps.pos;
    ps.inertialPos = (ps.pos + h * ps.vel).rowwise() + h * h * gravity.transpose();
    for (int i = 0; i < ps.pos.rows(); ++i) {
        if (ps.invMass[i] == 0) ps.inertialPos.row(i) = ps.pos.row(i);
    }
    ps.pos = ps.inertialPos;
}

void Solver::inertialGradient(Eigen::MatrixXd& gradient) const {
    auto& ps = particles;

    // M / h2 * (x - y)
    for (int i = 0; i < ps.pos.rows(); ++i) {
        if (ps.invMass[i] == 0) continue;
        gradient.row(i) += (ps.pos.row(i) - ps.inertialPos.row(i)) / (ps.invMass[i] * cache.factorizedH * cache.factorizedH);
    }
}

void Solver::energyGradient(Eigen::MatrixXd& gradient) const {
    auto& ps = particles;

    // grad U(x)
    for (const auto& e : energies) {
        Eigen::RowVector3d springGradient = Eigen::RowVector3d::Zero();
        springEnergyGradient(
            ps.pos.row(e.i) - ps.pos.row(e.j),
            e.restLength,
            e.stiffness,
            ps.prevPos.row(e.i) - ps.prevPos.row(e.j),
            springGradient
        );
        if (ps.invMass[e.i] > 0) gradient.row(e.i) += springGradient;
        if (ps.invMass[e.j] > 0) gradient.row(e.j) -= springGradient;
    }
}

void Solver::objectiveGradient(Eigen::MatrixXd& gradient) const {
    gradient.setZero(particles.pos.rows(), 3);
    inertialGradient(gradient);
    energyGradient(gradient);
}

void Solver::updateVelocities(double h) {
    auto& ps = particles;
    ps.vel = (ps.pos - ps.prevPos) / h;
}

void Solver::step() {
    auto& ps = particles;
    const double h = cache.factorizedH;

    for (int substep = 0; substep < substeps; ++substep) {
        predictPositions(h);

        Eigen::MatrixXd gradient;
        objectiveGradient(gradient);
        const Eigen::MatrixXd dx = cache.factor.solve(-gradient);
        ps.pos += dx;

        updateVelocities(h);
    }
}

} // namespace pd
