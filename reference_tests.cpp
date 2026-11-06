#include <Eigen/LU>
#include "core/bw.hpp"
#include "core/xpbd_newton.hpp"
#include "core/mpm.hpp"

#include "tests/scene_fixtures.hpp"
#include <cstdio>
#include <type_traits>

static int failures = 0;
static void check(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

template<class Solver>
static Solver spring() {
    auto cloth = makeReferenceCloth<Solver>(1, 2, 1, 1);
    cloth.particles = {{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d::Zero(), Eigen::Vector3d(0.2, 0, 0), 1},
                       {Eigen::Vector3d(2, 0, 0), Eigen::Vector3d::Zero(), Eigen::Vector3d(-0.7, 0, 0), 3}};
    cloth.stretchConstraints = {{0, 1, 1, 0.01}};
    cloth.bendConstraints.clear();
    cloth.substeps = 1;
    cloth.gravity.setZero();
    cloth.rebuildConnectivity();
    return cloth;
}

template<class Solver>
static void analyticSpring(const char* name) {
    auto cloth = spring<Solver>();
    cloth.step(0.1);
    const double length = (1.91 + 4) / 5, correction = (1.91 - length) / 4;
    const double error = std::max((cloth.particles[0].pos - Eigen::Vector3d(0.02 + correction, 0, 0)).norm(),
                                  (cloth.particles[1].pos - Eigen::Vector3d(1.93 - 3 * correction, 0, 0)).norm());
    std::printf("%s analytic implicit spring error %.3g\n", name, error);
    check(error < 1e-9, "unequal-mass implicit spring with incoming velocities");
}

static void bwLinearization() {
    bw::Solver cloth;
    cloth.particles = {
        {{0, 0, 0}, Eigen::Vector3d::Zero(), {0.2, 0, 0}, 1},
        {{2, 0, 0}, Eigen::Vector3d::Zero(), {-0.7, 0, 0}, 3}
    };
    cloth.energies = {{0, 1, 1, 100}};
    cloth.substeps = 1;
    cloth.gravity.setZero();
    cloth.particles[1].pos = Eigen::Vector3d(0.5, 0.2, -0.3);
    cloth.particles[0].vel = Eigen::Vector3d(0.2, -0.3, 0.1);
    cloth.particles[1].vel = Eigen::Vector3d(-0.1, 0.4, 0.2);
    constexpr double h = 0.03;
    Eigen::Matrix<double, 6, 1> x, velocity;
    x << cloth.particles[0].pos, cloth.particles[1].pos;
    velocity << cloth.particles[0].vel, cloth.particles[1].vel;
    const auto force = [](const Eigen::Matrix<double, 6, 1>& positions) {
        const Eigen::Vector3d d = positions.head<3>() - positions.tail<3>();
        const Eigen::Vector3d f = -100 * (d.norm() - 1) * d.normalized();
        Eigen::Matrix<double, 6, 1> out;
        out << f, -f;
        return out;
    };
    Eigen::Matrix<double, 6, 6> mass = Eigen::Matrix<double, 6, 6>::Identity(), jacobian;
    mass.bottomRightCorner<3, 3>() /= 3;
    for (int i = 0; i < 6; ++i) {
        auto plus = x, minus = x;
        plus[i] += 1e-6; minus[i] -= 1e-6;
        jacobian.col(i) = (force(plus) - force(minus)) / 2e-6;
    }
    const Eigen::Matrix<double, 6, 1> nextVelocity = (mass - h * h * jacobian).fullPivLu().solve(mass * velocity + h * force(x));
    const Eigen::Matrix<double, 6, 1> expected = x + h * nextVelocity;
    cloth.dt = h;
    cloth.step();
    check((cloth.particles[0].pos - expected.head<3>()).norm() < 1e-9 &&
          (cloth.particles[1].pos - expected.tail<3>()).norm() < 1e-9,
          "BW exact compressed-spring linearization agrees with finite-difference force Jacobian");
}

static void newtonChecks() {
    auto cloth = spring<XpbdNewton>();
    cloth.particles[0].invMass = 0;
    cloth.particles[0].pos.setZero();
    cloth.particles[1].pos = Eigen::Vector3d(1.4, 0.5, 0.2);
    cloth.particles[1].vel = Eigen::Vector3d(-0.2, 0.8, 0.3);
    const Eigen::Vector3d y = cloth.particles[1].pos + 0.02 * cloth.particles[1].vel;
    const double a = 0.02 * 0.02 * 100 * 3;
    const Eigen::Vector3d expected = y.normalized() * ((y.norm() + a) / (1 + a));
    cloth.step(0.02);
    check(cloth.converged && (cloth.particles[1].pos - expected).norm() < 1e-8,
          "nonlinear Newton agrees with analytic oblique pinned spring");
    check(cloth.particles[0].pos.isZero(), "Newton eliminates pinned DOFs exactly");
    check(cloth.primalResidual <= cloth.tolerance && cloth.constitutiveResidual <= cloth.tolerance,
          "Newton checks BOTH residuals");
    // Force constitutive balance while violating momentum balance.
    cloth.targets[1] += Eigen::Vector3d(0.1, -0.1, 0.03);
    const auto before = cloth.residual(0.02);
    check(before.constitutiveMax < 1e-8 && before.primalMax > 0.01, "constitutive balance alone is insufficient");
    cloth.solve(0.02);
    check(cloth.converged && cloth.primalResidual < 1e-8, "Newton repairs nonzero primal residual");
    cloth.iterations = 1;
    cloth.targets[1] += Eigen::Vector3d(0.3, 0.7, -0.2);
    cloth.solve(0.02);
    check(!cloth.converged, "Newton reports an insufficient nonlinear iteration budget");
    cloth.stretchConstraints[0].compliance = 0;
    bool rejected = false;
    try { cloth.step(0.02); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "Newton finite-compliance reduction rejects hard constraints");
}

template<class Solver>
static void clothChecks(const char* name) {
    auto cloth = makeReferenceCloth<Solver>(5, 7, 0.18, 0.01);
    cloth.particles[17].pos.z() += 0.05;
    const Eigen::Vector3d pin = cloth.particles[0].pos;
    int unconverged = 0;
    for (int i = 0; i < 90; ++i) {
        if (i == 20) cloth.particles[17].invMass = 0;
        if (i == 25) cloth.particles[17].invMass = 100;
        if (i == 50) {
            cloth.substeps = 2;
            for (auto& c : cloth.stretchConstraints) c.compliance = 1e-4;
        }
        cloth.step(1.0 / 60);
        check(cloth.particles[0].pos == pin, "reference solver preserves pins");
        for (const auto& p : cloth.particles)
            check(p.pos.allFinite() && p.vel.allFinite() && p.pos.norm() < 100, "reference cloth remains bounded and finite");
        if constexpr (std::is_same_v<Solver, XpbdNewton>) unconverged += !cloth.converged;
    }
    if constexpr (std::is_same_v<Solver, XpbdNewton>) check(unconverged == 0, "small edited cloth Newton convergence");
    std::printf("%s 90-frame edited 3D cloth passed\n", name);
}

static void mpmTransfers() {
    MpmSimulation simulation = makeMpmScene();
    simulation.gravity.setZero();
    double mass = 0;
    double angularMomentum = 0;
    MpmVec momentum = MpmVec::Zero();
    for (auto& p : simulation.particles) {
        p.affine << 0.1, -0.3, 0.2, -0.1;
        mass += p.mass;
        momentum += p.mass * p.vel;
        angularMomentum += p.mass * (p.pos.x() * p.vel.y() - p.pos.y() * p.vel.x()
            + simulation.dx * simulation.dx / 4 * (p.affine(1, 0) - p.affine(0, 1)));
    }
    simulation.particleToGrid(1e-4);
    double gridMass = 0;
    double gridAngularMomentum = 0;
    MpmVec gridMomentum = MpmVec::Zero();
    for (const auto& node : simulation.grid) { gridMass += node.mass; gridMomentum += node.momentum; }
    for (int i = 0; i <= simulation.resolution; ++i) for (int j = 0; j <= simulation.resolution; ++j) {
        const auto& node = simulation.node(i, j);
        gridAngularMomentum += simulation.dx * (i * node.momentum.y() - j * node.momentum.x());
    }
    check(std::abs(gridMass - mass) < 1e-12 && (gridMomentum - momentum).norm() < 1e-12,
          "MPM P2G conserves mass and linear momentum");
    check(std::abs(gridAngularMomentum - angularMomentum) < 1e-12, "APIC P2G preserves angular momentum");
    simulation.updateGrid(1e-4, false);
    simulation.gridToParticle(1e-4);
    MpmVec recoveredMomentum = MpmVec::Zero();
    for (const auto& p : simulation.particles) recoveredMomentum += p.mass * p.vel;
    check((recoveredMomentum - momentum).norm() < 1e-12, "MPM full transfer preserves linear momentum without external forces");
    MpmParticle original = simulation.particles.front();
    original.deformation.setIdentity();
    simulation.particles = {original};
    simulation.particleToGrid(1e-4);
    simulation.updateGrid(1e-4, false);
    simulation.gridToParticle(1e-4);
    check((simulation.particles[0].vel - original.vel).norm() < 1e-12 &&
          (simulation.particles[0].affine - original.affine).norm() < 1e-10,
          "APIC reproduces an affine velocity field");
    MpmMat rotation;
    const double angle = 0.7;
    rotation << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
    check(simulation.stress(rotation).norm() < 1e-10, "fixed-corotated stress vanishes under rigid rotation");
    simulation = makeMpmScene();
    simulation.gravity.setZero();
    for (auto& p : simulation.particles) p.vel.setZero();
    const MpmVec initial = simulation.particles.front().pos;
    simulation.step(0.02);
    check((simulation.particles.front().pos - initial).norm() < 1e-12 &&
          (simulation.particles.front().deformation - MpmMat::Identity()).norm() < 1e-12,
          "MPM undeformed rest state");
}

static void mpmEvolution() {
    MpmSimulation simulation = makeMpmScene();
    double minJ = 1, maxJ = 1;
    for (int frame = 0; frame < 120; ++frame) {
        if (frame == 60) simulation.young = 800;
        simulation.step(1.0 / 60);
        for (const auto& p : simulation.particles) {
            const double J = p.deformation.determinant();
            minJ = std::min(minJ, J); maxJ = std::max(maxJ, J);
            check(p.pos.allFinite() && p.vel.allFinite() && J > 0 &&
                  (p.pos.array() > 0).all() && (p.pos.array() < 1).all(), "MPM bounded orientation-preserving evolution");
        }
    }
    std::printf("MPM two-second elastic block drop, J range [%.6g, %.6g]\n", minJ, maxJ);
}

int main() {
    try {
        analyticSpring<XpbdNewton>("XPBD Newton");
        bwLinearization();
        newtonChecks();
        clothChecks<XpbdNewton>("XPBD Newton");
        mpmTransfers();
        mpmEvolution();
    } catch (const std::exception& error) { std::fprintf(stderr, "Exception: %s\n", error.what()); return 1; }
    if (failures) std::fprintf(stderr, "%d checks failed\n", failures);
    else std::puts("All BW/Newton/MPM checks passed");
    return failures ? 1 : 0;
}
