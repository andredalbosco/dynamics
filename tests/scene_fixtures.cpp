#include "scene_fixtures.hpp"

#include <map>
#include <random>

xpbd::Solver makeSpring() {
    xpbd::Solver scene;
    scene.gravity.setZero();
    scene.particles = {
        {{-1, 2.5, 0}, {-1, 2.5, 0}, Eigen::Vector3d::Zero(), 0},
        {{0.5, 2.5, 0}, {0.5, 2.5, 0}, Eigen::Vector3d::Zero(), 1}
    };
    scene.constraints = {{0, 1, 1, 0.01}};
    return scene;
}

xpbd::Solver makePendulum() {
    xpbd::Solver scene;
    const Eigen::Vector3d anchor {0, 3, 0};
    const Eigen::Vector3d bob = anchor + Eigen::Vector3d(std::sin(0.65), -std::cos(0.65), 0);
    scene.particles = {
        {anchor, anchor, Eigen::Vector3d::Zero(), 0},
        {bob, bob, Eigen::Vector3d::Zero(), 1}
    };
    scene.constraints = {{0, 1, 1, 0}};
    return scene;
}

xpbd::Solver makeDoublePendulum() {
    xpbd::Solver scene;
    const Eigen::Vector3d anchor {0, 3, 0};
    const Eigen::Vector3d first = anchor + Eigen::Vector3d(std::sin(0.9), -std::cos(0.9), 0);
    const Eigen::Vector3d second = first + Eigen::Vector3d(std::sin(-0.4), -std::cos(-0.4), 0);
    scene.particles = {
        {anchor, anchor, Eigen::Vector3d::Zero(), 0},
        {first, first, Eigen::Vector3d::Zero(), 1},
        {second, second, Eigen::Vector3d::Zero(), 0.5}
    };
    scene.constraints = {{0, 1, 1, 0}, {1, 2, 1, 0}};
    return scene;
}

ClothFixture makeClothFixture(int rows, int cols, double spacing, double particleMass) {
    ClothFixture fixture;
    xpbd::Solver& state = fixture.simulation;
    std::vector<xpbd::DistanceConstraint> bending;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Eigen::Vector3d pos {(j - (cols - 1) * 0.5) * spacing, 4.5 - i * spacing, 0};
            const bool pinned = i == 0 && (j == 0 || j == cols - 1);
            state.particles.push_back({pos, pos, Eigen::Vector3d::Zero(), pinned ? 0 : 1.0 / particleMass});
        }
    }

    std::vector<int> diagonals((rows - 1) * (cols - 1), 0);
    std::fill_n(diagonals.begin(), diagonals.size() / 2, 1);
    std::mt19937 random {42};
    std::shuffle(diagonals.begin(), diagonals.end(), random);

    for (int i = 0; i + 1 < rows; ++i) {
        for (int j = 0; j + 1 < cols; ++j) {
            const int a = i * cols + j;
            const int b = a + 1;
            const int c = a + cols;
            const int d = c + 1;
            if (diagonals[i * (cols - 1) + j]) {
                fixture.triangles.push_back({a, c, d});
                fixture.triangles.push_back({a, d, b});
            } else {
                fixture.triangles.push_back({a, c, b});
                fixture.triangles.push_back({b, c, d});
            }
        }
    }

    auto makeConstraint = [&](int i, int j, double compliance = 0) {
        const double restLength = (state.particles[j].pos - state.particles[i].pos).norm();
        return xpbd::DistanceConstraint {i, j, restLength, compliance};
    };
    // Each edge stores the vertex opposite it in its first triangle.
    std::map<std::pair<int, int>, int> opposite;
    for (const auto& triangle : fixture.triangles) {
        for (int k = 0; k < 3; ++k) {
            const int i = triangle[k];
            const int j = triangle[(k + 1) % 3];
            const int across = triangle[(k + 2) % 3];
            const auto [edge, first] = opposite.emplace(std::minmax(i, j), across);
            if (first) {
                state.constraints.push_back(makeConstraint(i, j));
            } else {
                bending.push_back(makeConstraint(edge->second, across, 1));
            }
        }
    }
    fixture.stretchCount = state.constraints.size();
    state.constraints.insert(state.constraints.end(), bending.begin(), bending.end());
    return fixture;
}

xpbd::Solver makeCloth(int rows, int cols, double spacing, double particleMass) {
    return makeClothFixture(rows, cols, spacing, particleMass).simulation;
}

MpmSimulation makeMpmScene() {
    MpmSimulation simulation;
    for (int group = 0; group < 2; ++group) {
        const MpmVec center = group ? MpmVec(0.66, 0.48) : MpmVec(0.34, 0.74);
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 20; ++j) {
                MpmParticle p;
                p.pos = center + 0.5 * simulation.dx * MpmVec(i - 9.5, j - 9.5);
                p.vel = MpmVec(group ? -0.6 : 0.6, 0);
                p.volume = 0.25 * simulation.dx * simulation.dx; // Four particles per cell.
                p.mass = simulation.density * p.volume;
                p.group = group;
                simulation.particles.push_back(p);
            }
        }
    }
    return simulation;
}
