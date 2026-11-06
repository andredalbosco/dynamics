#include "xpbd_newton.hpp"

#include <Eigen/SparseLU>

XpbdNewton::Residual XpbdNewton::residual(double h) const {
    Residual r;
    r.primal.resize(particles.size(), Eigen::Vector3d::Zero());
    r.constitutive.resize(constraintCount());
    for (size_t i = 0; i < particles.size(); ++i)
        if (particles[i].invMass > 0)
            r.primal[i] = (particles[i].pos - targets[i]) / particles[i].invMass;
    for (int e = 0; e < constraintCount(); ++e) {
        const auto& c = constraint(e);
        const Eigen::Vector3d d = particles[c.i].pos - particles[c.j].pos;
        const double length = d.norm();
        if (length > 1e-12) {
            const Eigen::Vector3d force = c.lambda * d / length;
            if (particles[c.i].invMass > 0) r.primal[c.i] -= force;
            if (particles[c.j].invMass > 0) r.primal[c.j] += force;
        }
        const double q = length - c.restLength + c.compliance / (h * h) * c.lambda;
        r.constitutive[e] = q;
        r.constitutiveMax = std::max(r.constitutiveMax, std::abs(q));
        r.merit += q * q;
    }
    for (size_t i = 0; i < particles.size(); ++i) {
        // Scale momentum residuals to position units before globalization.
        const Eigen::Vector3d scaled = particles[i].invMass * r.primal[i];
        r.primalMax = std::max(r.primalMax, scaled.cwiseAbs().maxCoeff());
        r.merit += scaled.squaredNorm();
    }
    return r;
}

void XpbdNewton::solve(double h) {
    std::vector<int> free(particles.size(), -1);
    int count = 0;
    for (size_t i = 0; i < particles.size(); ++i)
        if (particles[i].invMass > 0) free[i] = 3 * count++;
    lastIterations = 0;
    for (int it = 0; it < iterations; ++it) {
        const Residual r = residual(h);
        if (std::max(r.primalMax, r.constitutiveMax) <= tolerance) break;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(3 * count);
        std::vector<Eigen::Triplet<double>> entries;
        for (size_t i = 0; i < particles.size(); ++i) {
            if (free[i] < 0) continue;
            rhs.segment<3>(free[i]) = -r.primal[i];
            for (int a = 0; a < 3; ++a)
                entries.emplace_back(free[i] + a, free[i] + a, 1 / particles[i].invMass);
        }
        for (int e = 0; e < constraintCount(); ++e) {
            const auto& c = constraint(e);
            const Eigen::Vector3d d = particles[c.i].pos - particles[c.j].pos;
            const double length = d.norm();
            if (length < 1e-12) continue; // Undefined gradient, same degeneracy as XPBD.
            const Eigen::Vector3d n = d / length;
            const double inverseAlpha = h * h / c.compliance;
            const Eigen::Matrix3d nn = n * n.transpose();
            const Eigen::Matrix3d block = inverseAlpha * nn - c.lambda / length * (Eigen::Matrix3d::Identity() - nn);
            for (int i : {c.i, c.j}) {
                if (free[i] < 0) continue;
                rhs.segment<3>(free[i]) -=
                    (i == c.i ? 1 : -1) * inverseAlpha * r.constitutive[e] * n;
                for (int j : {c.i, c.j}) {
                    if (free[j] < 0) continue;
                    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b)
                        entries.emplace_back(free[i] + a, free[j] + b,
                            (i == j ? 1 : -1) * block(a, b));
                }
            }
        }
        Eigen::VectorXd dx = Eigen::VectorXd::Zero(3 * count);
        if (count) {
            Eigen::SparseMatrix<double> matrix(3 * count, 3 * count);
            matrix.setFromTriplets(entries.begin(), entries.end());
            Eigen::SparseLU<Eigen::SparseMatrix<double>> factor;
            factor.compute(matrix);
            if (factor.info() != Eigen::Success)
                throw std::runtime_error("Newton factorization failed");
            dx = factor.solve(rhs);
            if (factor.info() != Eigen::Success || !dx.allFinite())
                throw std::runtime_error("Newton solve failed");
        }
        std::vector<Eigen::Vector3d> old;
        for (const auto& p : particles) old.push_back(p.pos);
        std::vector<double> lambda, dlambda;
        for (int e = 0; e < constraintCount(); ++e) {
            const auto& c = constraint(e);
            const Eigen::Vector3d d = old[c.i] - old[c.j];
            Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
            if (free[c.i] >= 0) displacement += dx.segment<3>(free[c.i]);
            if (free[c.j] >= 0) displacement -= dx.segment<3>(free[c.j]);
            const double Jdx = d.norm() > 1e-12 ? d.normalized().dot(displacement) : 0;
            lambda.push_back(c.lambda);
            dlambda.push_back(-(r.constitutive[e] + Jdx) * h * h / c.compliance);
        }
        bool accepted = false;
        double step = 1;
        for (int ls = 0; ls < 24; ++ls, step *= 0.5) {
            for (size_t i = 0; i < particles.size(); ++i)
                if (free[i] >= 0) particles[i].pos = old[i] + step * dx.segment<3>(free[i]);
            for (int e = 0; e < constraintCount(); ++e)
                constraint(e).lambda = lambda[e] + step * dlambda[e];
            const Residual trial = residual(h);
            if (std::isfinite(trial.merit) && trial.merit <= (1 - 1e-4 * step) * r.merit) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            for (size_t i = 0; i < particles.size(); ++i) particles[i].pos = old[i];
            for (int e = 0; e < constraintCount(); ++e) constraint(e).lambda = lambda[e];
            break;
        }
        ++lastIterations;
    }
    const Residual r = residual(h);
    primalResidual = r.primalMax;
    constitutiveResidual = r.constitutiveMax;
    converged = std::max(primalResidual, constitutiveResidual) <= tolerance;
}

void XpbdNewton::step(double dt) {
    const double h = substepSize(dt);
    for (int e = 0; e < constraintCount(); ++e) stiffness(constraint(e));
    for (int substep = 0; substep < substeps; ++substep) {
        predict(h);
        for (int e = 0; e < constraintCount(); ++e) constraint(e).lambda = 0;
        solve(h);
        updateVelocities(h);
    }
}
