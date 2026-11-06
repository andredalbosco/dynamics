#include "core/pbd.hpp"

#include "tests/scene_fixtures.hpp"
#include <cstdio>

static int failures = 0;
static void check(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

template<class Solver>
static Solver pair(double compliance = 0.01) {
    auto cloth = makeReferenceCloth<Solver>(1, 2, 1, 1);
    cloth.particles = {{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), Eigen::Vector3d::Zero(), 1},
                       {Eigen::Vector3d(2, 0, 0), Eigen::Vector3d(2, 0, 0), Eigen::Vector3d::Zero(), 3}};
    cloth.stretchConstraints = {{0, 1, 1, compliance}};
    cloth.bendConstraints.clear();
    cloth.gravity.setZero();
    cloth.substeps = 1;
    cloth.iterations = 400;
    cloth.rebuildConnectivity();
    return cloth;
}

static void pbdChecks() {
    auto cloth = pair<Pbd>();
    cloth.step(0.1);
    check((cloth.particles[0].pos - Eigen::Vector3d(0.25, 0, 0)).norm() < 1e-12 &&
          (cloth.particles[1].pos - Eigen::Vector3d(1.25, 0, 0)).norm() < 1e-12, "PBD mass-weighted hard projection");
    auto a = pair<Pbd>(), b = a;
    a.stretchStiffness = b.stretchStiffness = 0.6;
    a.iterations = 1;
    b.iterations = 20;
    a.step(0.1); b.step(0.1);
    check((a.particles[0].pos - b.particles[0].pos).norm() < 1e-12, "isolated PBD stiffness normalized over sweeps");
    auto pinned = pair<Pbd>();
    pinned.particles[0].invMass = 0;
    pinned.step(0.1);
    check(pinned.particles[0].pos.isZero() && std::abs(pinned.particles[1].pos.x() - 1) < 1e-12,
          "PBD exact pin");
}

template<class Solver>
static void edgeCases() {
    auto falling = pair<Solver>();
    falling.stretchConstraints.clear();
    falling.rebuildConnectivity();
    falling.gravity = Eigen::Vector3d(0, -9.81, 0);
    falling.step(0.1);
    check(std::abs(falling.particles[0].pos.y() + 0.0981) < 1e-12 &&
          std::abs(falling.particles[0].vel.y() + 0.981) < 1e-12, "unconstrained gravity integration");
    auto cloth = pair<Solver>();
    cloth.particles[1].pos.setZero();
    cloth.iterations = 2;
    cloth.step(0.1);
    for (const auto& p : cloth.particles) check(p.pos.allFinite() && p.vel.allFinite(), "coincident endpoints remain finite");
    for (auto& p : cloth.particles) p.invMass = 0;
    const Eigen::Vector3d a = cloth.particles[0].pos, b = cloth.particles[1].pos;
    cloth.step(0.1);
    check(cloth.particles[0].pos == a && cloth.particles[1].pos == b, "all-fixed system");
}

template<class Solver>
static void evolve(const char* name) {
    auto cloth = makeReferenceCloth<Solver>(17, 21, 0.18, 0.01);
    const auto original = makeClothFixture(17, 21, 0.18, 0.01);
    check(original.triangles.size() == 640 && original.simulation.constraints.size() == 1920 &&
          original.stretchCount == 996 && cloth.stretchConstraints.size() == 996 &&
          cloth.bendConstraints.size() == 924, "shared scene topology");
    const Eigen::Vector3d left = cloth.particles[0].pos, right = cloth.particles[20].pos;
    cloth.particles[170].pos.z() = 0.15;
    double maxStretch = 0;
    for (int frame = 0; frame < 180; ++frame) {
        if (frame == 60) {
            cloth.particles[170].pos += Eigen::Vector3d(0.1, 0.1, 0.2);
            cloth.particles[170].invMass = 0;
        }
        if (frame == 70) cloth.particles[170].invMass = 100;
        if (frame == 100) {
            cloth.substeps = 2;
            for (auto& c : cloth.stretchConstraints) c.compliance = 1e-4;
        }
        const Eigen::Vector3d grab = cloth.particles[170].pos;
        cloth.step(1.0 / 60);
        if (frame >= 60 && frame < 70) check(cloth.particles[170].pos == grab, "grabbed particle remains fixed");
        check(cloth.particles[0].pos == left && cloth.particles[20].pos == right, "cloth corner pins remain exact");
        for (const auto& p : cloth.particles)
            check(p.pos.allFinite() && p.vel.allFinite() && p.pos.norm() < 100 && p.vel.norm() < 1e4,
                  "bounded finite cloth evolution");
        for (const auto& c : cloth.stretchConstraints)
            maxStretch = std::max(maxStretch, std::abs((cloth.particles[c.i].pos - cloth.particles[c.j].pos).norm() / c.restLength - 1));
    }
    std::printf("%s 180-frame edited/grabbed cloth peak relative stretch: %.6g\n", name, maxStretch);
}

int main() {
    pbdChecks();
    edgeCases<Pbd>();
    evolve<Pbd>("PBD");
    if (failures) std::fprintf(stderr, "%d checks failed\n", failures);
    else std::puts("All CPU solver checks passed");
    return failures ? 1 : 0;
}
