#include "core/vbd.hpp"

#include <cmath>
#include <cstdio>

bool checkDerivatives() {
    vbd::SpringEnergy energy {0, 1, 1, 73};
    std::vector<vbd::Particle> particles(3);
    const Eigen::Vector3d origin {0.3, -0.2, 0.4};
    particles[1].pos = origin;

    for (const Eigen::Vector3d& displacement : {
        Eigen::Vector3d(0.2, -0.3, 0.1), Eigen::Vector3d(1.4, 0.8, -0.6), Eigen::Vector3d(0, 1, 0)}) {
        particles[0].pos = origin + displacement;
        Eigen::Vector3d gradients[2];

        for (int vertex : {0, 1}) {
            Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
            Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
            vbd::springEnergyDerivatives(particles[vertex].pos - particles[1 - vertex].pos, energy.restLength, energy.stiffness, gradient, hessian);
            gradients[vertex] = gradient;

            // Perturb actual endpoint positions: this checks each endpoint's sign and Hessian block.
            constexpr double epsilon = 1e-6;
            for (int axis = 0; axis < 3; ++axis) {
                auto plus = particles;
                auto minus = particles;
                plus[vertex].pos[axis] += epsilon;
                minus[vertex].pos[axis] -= epsilon;
                const double derivative = (vbd::springEnergyValue(plus[0].pos - plus[1].pos, energy.restLength, energy.stiffness) - vbd::springEnergyValue(minus[0].pos - minus[1].pos, energy.restLength, energy.stiffness)) / (2 * epsilon);
                if (std::abs(derivative - gradient[axis]) > 1e-6) return false;

                Eigen::Vector3d gp = Eigen::Vector3d::Zero();
                Eigen::Vector3d gm = Eigen::Vector3d::Zero();
                Eigen::Matrix3d unused = Eigen::Matrix3d::Zero();
                vbd::springEnergyDerivatives(plus[vertex].pos - plus[1 - vertex].pos, energy.restLength, energy.stiffness, gp, unused);
                unused.setZero();
                vbd::springEnergyDerivatives(minus[vertex].pos - minus[1 - vertex].pos, energy.restLength, energy.stiffness, gm, unused);
                if (((gp - gm) / (2 * epsilon) - hessian.col(axis)).norm() > 1e-5) return false;
            }

            Eigen::Vector3d accumulated = Eigen::Vector3d::Ones();
            Eigen::Matrix3d assembled = Eigen::Matrix3d::Identity();
            vbd::springEnergyDerivatives(particles[vertex].pos - particles[1 - vertex].pos, energy.restLength, energy.stiffness, accumulated, assembled);
            if ((accumulated - gradient - Eigen::Vector3d::Ones()).norm() > 1e-12 ||
                (assembled - hessian - Eigen::Matrix3d::Identity()).norm() > 1e-12) return false;

            auto trialParticles = particles;
            const Eigen::Vector3d trial = particles[vertex].pos + Eigen::Vector3d(0.1, -0.2, 0.3);
            trialParticles[vertex].pos = trial;
            if (std::abs(vbd::springEnergyValue(trial - particles[1 - vertex].pos, energy.restLength, energy.stiffness) - vbd::springEnergyValue(trialParticles[0].pos - trialParticles[1].pos, energy.restLength, energy.stiffness)) > 1e-12)
                return false;
        }
        if ((gradients[0] + gradients[1]).norm() > 1e-12) return false;

        if (std::abs(vbd::springEnergyValue(particles[0].pos - particles[1].pos, energy.restLength, energy.stiffness) -
            vbd::springEnergyValue(particles[1].pos - particles[0].pos, energy.restLength, energy.stiffness)) > 1e-12) return false;
    }
    return true;
}

bool checkValues() {
    // Connectivity indices are irrelevant to the mathematical functions.
    vbd::SpringEnergy energy {7, 19, 1, 100};
    const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    if (vbd::springEnergyValue(Eigen::Vector3d(2, 0, 0), energy.restLength, energy.stiffness) != 50) return false;
    if (vbd::springEnergyValue(Eigen::Vector3d(1, 0, 0), energy.restLength, energy.stiffness) != 0) return false;

    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
    vbd::springEnergyDerivatives(origin, energy.restLength, energy.stiffness, gradient, hessian);
    return gradient.allFinite() && hessian.allFinite() && vbd::springEnergyValue(origin, energy.restLength, energy.stiffness) == 50;
}

int main() {
    if (!checkValues() || !checkDerivatives()) {
        std::fputs("Spring energy checks failed\n", stderr);
        return 1;
    }
    std::puts("Spring energy: values, both endpoint gradients/Hessians, assembly, and trial evaluation passed");
    return 0;
}
