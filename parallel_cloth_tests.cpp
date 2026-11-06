#include "core/parallel_cloth.hpp"
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

static int failures = 0;
static void check(bool good, const char* message) {
    if (!good) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

static void topology() {
    ParallelCloth cloth(5, 7);
    check(cloth.numX == 6 && cloth.numY == 8, "odd dimensions rounded up like original");
    check(cloth.particles.size() == 63 && cloth.triangles.size() == 96, "rectangular mesh dimensions");
    check(cloth.links.size() == 300, "stretch, shear, and two-hop bending counts");
    for (int pass = 0; pass < 4; ++pass) {
        std::set<int> used;
        for (int e = cloth.passOffsets[pass]; e < cloth.passOffsets[pass + 1]; ++e) {
            check(used.insert(cloth.links[e].a).second, "color pass has no repeated endpoint A");
            check(used.insert(cloth.links[e].b).second, "color pass has no repeated endpoint B");
        }
    }
    check(cloth.maxRelativeError() == 0, "rest lengths match initial cloth");
    cloth.updateNormals();
    for (const auto& n : cloth.normals)
        check((n + ParallelCloth::Vec::UnitY()).norm() < 1e-6f, "initial triangle winding and unit normals");
    bool rejected = false;
    try { ParallelCloth invalid(0, 4); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid scene rejected");
}

static void contacts() {
    ParallelCloth c(2, 2);
    c.gravity.setZero();
    auto& p = c.particles[0];
    p.pos = {0.1f, 1.5f, 0};
    p.velocity = {0, 0, 0};
    c.integrate(0.01f);
    check(std::abs((p.pos - c.sphereCenter).norm() - c.sphereRadius - c.thickness) < 1e-6f,
        "sphere projection has configured thickness");
    p.pos = {0, -1, 0};
    p.velocity = {1, 0, 0};
    c.friction = 0.25f;
    c.integrate(0.1f);
    check(std::abs(p.pos.y() - c.thickness) < 1e-7f && std::abs(p.pos.x() - 0.075f) < 1e-6f,
        "ground projection retains original positional friction");
    p.pos = {0.1f, 1.5f, 0};
    p.invMass = 0;
    const auto pinned = p.pos;
    c.integrate(0.1f);
    check((p.pos - pinned).norm() == 0, "pinned particles skip integration and contact");
    c.reset();
    check(c.particles[0].invMass == 1 && c.particles[0].velocity.isZero() &&
        (c.particles[0].pos - c.restPositions[0]).norm() == 0, "reset restores masses, positions, and velocities");
}

static void projection() {
    ParallelCloth c(2, 2);
    c.particles[0].pos = {0, 0, 0}; c.particles[0].invMass = 1;
    c.particles[1].pos = {2, 0, 0}; c.particles[1].invMass = 3;
    c.links = {{0, 1, 1}};
    c.solveRange(0, 1, false);
    check(std::abs(c.particles[0].pos.x() - 0.25f) < 1e-7f &&
        std::abs(c.particles[1].pos.x() - 1.25f) < 1e-7f, "mass-weighted pair projection");
    c.particles[0].pos = c.particles[1].pos;
    c.solveRange(0, 1, true);
    check(c.particles[0].pos.allFinite(), "coincident pair remains finite");
}

static void gatherEquivalence() {
    ParallelCloth scatter(8, 6), gather = scatter;
    scatter.mode = ParallelCloth::Mode::Jacobi;
    gather.mode = ParallelCloth::Mode::Gather;
    scatter.gravity.setZero(); gather.gravity.setZero();
    scatter.sphereEnabled = gather.sphereEnabled = false;
    scatter.groundEnabled = gather.groundEnabled = false;
    scatter.particles[12].pos += ParallelCloth::Vec(0.002f, 0.003f, -0.002f);
    gather.particles[12].pos = scatter.particles[12].pos;
    scatter.particles[4].invMass = gather.particles[4].invMass = 0;
    scatter.substeps = gather.substeps = 2;
    for (int frame = 0; frame < 3; ++frame) {
        scatter.step(1.0f / 60); gather.step(1.0f / 60);
    }
    float maximum = 0;
    for (size_t i = 0; i < scatter.particles.size(); ++i)
        maximum = std::max(maximum, (scatter.particles[i].pos - gather.particles[i].pos).norm());
    std::printf("scatter/gather maximum position difference %.9g\n", maximum);
    check(maximum < 2e-6f, "Jacobi scatter and recomputing gather agree for deformed pinned cloth");
}

static void trajectory(ParallelCloth::Mode mode) {
    ParallelCloth c(8, 8, 0.1f);
    c.mode = mode;
    const auto initial = c.particles[0].pos;
    c.particles[0].invMass = 0;
    for (int frame = 0; frame < 120; ++frame) c.step(1.0f / 60);
    c.updateNormals();
    float movement = 0;
    for (size_t i = 0; i < c.particles.size(); ++i) {
        check(c.particles[i].pos.allFinite() && c.particles[i].velocity.allFinite() && c.normals[i].allFinite(),
            "finite two-second contacting trajectory");
        movement += (c.particles[i].pos - c.restPositions[i]).norm();
    }
    check((c.particles[0].pos - initial).norm() == 0, "solver preserves pinned endpoint");
    check(movement > 1, "cloth progresses");
    c.substeps = 0;
    bool rejected = false;
    try { c.step(1.0f / 60); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid substep count rejected");
}

// Export float positions/normals for the archived Warp/Taichi comparison script.
int main(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "--dump") {
        const int mode = std::stoi(argv[2]), frames = std::stoi(argv[3]), size = std::stoi(argv[4]);
        if (mode < 0 || mode > 3 || frames < 0 || frames > 10000) return 2;
        ParallelCloth c(size, size);
        c.mode = static_cast<ParallelCloth::Mode>(mode);
        for (int i = 0; i < frames; ++i) c.step(1.0f / 60);
        c.updateNormals();
        std::ofstream out(argv[5]);
        out << std::setprecision(9);
        for (size_t i = 0; i < c.particles.size(); ++i)
            out << c.particles[i].pos.transpose() << ' ' << c.normals[i].transpose() << '\n';
        return out ? 0 : 1;
    }
    if (argc != 1) return 2;
    topology(); contacts(); projection(); gatherEquivalence();
    for (int mode = 0; mode < 4; ++mode) trajectory(static_cast<ParallelCloth::Mode>(mode));
    std::printf("parallel cloth: %d failures\n", failures);
    return failures ? 1 : 0;
}
