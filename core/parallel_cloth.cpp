#include "parallel_cloth.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>

ParallelCloth::ParallelCloth(int cellsX, int cellsY, float spacing, float height)
    : numX(cellsX), numY(cellsY) {
    if (cellsX < 2 || cellsY < 2 || cellsX > 1024 || cellsY > 1024 ||
        !(spacing > 0) || !std::isfinite(spacing) || !std::isfinite(height))
        throw std::invalid_argument("Cloth requires 2..1024 cells and finite positive spacing");
    // The original rounds odd cell counts up for its two-color stretch passes.
    numX += numX % 2;
    numY += numY % 2;
    for (int x = 0; x <= numX; ++x) {
        for (int y = 0; y <= numY; ++y) {
            Vec p {(-numX * 0.5f + x) * spacing, height, (-numY * 0.5f + y) * spacing};
            particles.push_back({p, p});
            restPositions.push_back(p);
        }
    }
    auto link = [&](int a, int b) {
        links.push_back({a, b, (particles[b].pos - particles[a].pos).norm()});
    };
    int pass = 0;
    for (int color = 0; color < 2; ++color) {
        for (int x = 0; x <= numX; ++x)
            for (int y = 0; y < numY / 2; ++y) {
                const int a = x * (numY + 1) + 2 * y + color;
                link(a, a + 1);
            }
        passOffsets[++pass] = int(links.size());
    }
    for (int color = 0; color < 2; ++color) {
        for (int x = 0; x < numX / 2; ++x)
            for (int y = 0; y <= numY; ++y) {
                const int a = (2 * x + color) * (numY + 1) + y;
                link(a, a + numY + 1);
            }
        passOffsets[++pass] = int(links.size());
    }
    for (int x = 0; x < numX; ++x)
        for (int y = 0; y < numY; ++y) {
            const int a = x * (numY + 1) + y, b = a + numY + 1;
            link(a, b + 1);
            link(b, a + 1);
            triangles.emplace_back(a, b, b + 1);
            triangles.emplace_back(a, b + 1, a + 1);
        }
    for (int x = 0; x <= numX; ++x)
        for (int y = 0; y < numY - 1; ++y) {
            const int a = x * (numY + 1) + y;
            link(a, a + 2);
        }
    for (int x = 0; x < numX - 1; ++x)
        for (int y = 0; y <= numY; ++y) {
            const int a = x * (numY + 1) + y;
            link(a, a + 2 * (numY + 1));
        }
    passOffsets[5] = int(links.size());
    corrections.resize(particles.size(), Vec::Zero());
    normals.resize(particles.size(), Vec::UnitY());
    incident.resize(particles.size());
    for (int e = 0; e < int(links.size()); ++e) {
        incident[links[e].a].push_back(e);
        incident[links[e].b].push_back(e);
    }
}

void ParallelCloth::reset() {
    for (size_t i = 0; i < particles.size(); ++i)
        particles[i] = {restPositions[i], restPositions[i]};
    std::fill(corrections.begin(), corrections.end(), Vec::Zero());
    updateNormals();
}

void ParallelCloth::integrate(float h) {
    for (auto& p : particles) {
        p.previous = p.pos;
        if (p.invMass == 0) continue;
        p.velocity += gravity * h;
        p.pos += p.velocity * h;
        if (sphereEnabled && (p.pos - sphereCenter).norm() < sphereRadius + thickness) {
            const Vec mixed = p.pos * (1 - friction) + p.previous * friction;
            const Vec r = mixed - sphereCenter;
            const float length = r.norm();
            if (length > 1e-8f)
                p.pos = sphereCenter + r * ((sphereRadius + thickness) / length);
        }
        if (groundEnabled && p.pos.y() < thickness) {
            const Vec mixed = p.pos * (1 - friction) + p.previous * friction;
            p.pos = {mixed.x(), thickness, mixed.z()};
        }
    }
}

void ParallelCloth::solveRange(int first, int last, bool jacobi) {
    if (jacobi) std::fill(corrections.begin(), corrections.end(), Vec::Zero());
    for (int e = first; e < last; ++e) {
        const auto& c = links[e];
        auto& a = particles[c.a];
        auto& b = particles[c.b];
        const float w = a.invMass + b.invMass;
        const Vec delta = b.pos - a.pos;
        const float length = delta.norm();
        if (w == 0 || length <= 1e-8f) continue;
        const Vec dp = (delta / length) * ((length - c.rest) / w);
        if (jacobi) {
            corrections[c.a] += a.invMass * dp;
            corrections[c.b] -= b.invMass * dp;
        } else {
            a.pos += a.invMass * dp;
            b.pos -= b.invMass * dp;
        }
    }
    if (jacobi)
        for (size_t i = 0; i < particles.size(); ++i)
            particles[i].pos += corrections[i] * jacobiScale;
}

void ParallelCloth::solveGather() {
    // Same hard-constraint Jacobi sum, gathered per particle as in the WGSL port.
    // Read all old positions before applying any corrections.
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& a = particles[i];
        Vec correction = Vec::Zero();
        for (int e : incident[i]) {
            const auto& c = links[e];
            const auto& b = particles[c.a == int(i) ? c.b : c.a];
            const float w = a.invMass + b.invMass;
            const Vec delta = b.pos - a.pos;
            const float length = delta.norm();
            if (w != 0 && length > 1e-8f)
                correction += (delta / length) * ((length - c.rest) / w) * a.invMass;
        }
        corrections[i] = correction;
    }
    for (size_t i = 0; i < particles.size(); ++i)
        particles[i].pos += corrections[i] * jacobiScale;
}

void ParallelCloth::step(float dt) {
    if (!(dt > 0) || !std::isfinite(dt) || substeps < 1 ||
        !std::isfinite(jacobiScale) || jacobiScale < 0 || jacobiScale > 1)
        throw std::invalid_argument("Positive timestep/substeps and Jacobi scale in [0,1] required");
    const float h = dt / substeps;
    for (int substep = 0; substep < substeps; ++substep) {
        integrate(h);
        switch (mode) {
            case Mode::Sequential: solveRange(0, int(links.size()), false); break;
            case Mode::Hybrid:
                for (int pass = 0; pass < 5; ++pass)
                    solveRange(passOffsets[pass], passOffsets[pass + 1], pass == 4);
                break;
            case Mode::Jacobi: solveRange(0, int(links.size()), true); break;
            case Mode::Gather: solveGather(); break;
        }
        for (auto& p : particles) p.velocity = (p.pos - p.previous) / h;
    }
}

void ParallelCloth::updateNormals() {
    std::fill(normals.begin(), normals.end(), Vec::Zero());
    for (const auto& t : triangles) {
        const Vec n = (particles[t[1]].pos - particles[t[0]].pos).cross(
            particles[t[2]].pos - particles[t[0]].pos);
        for (int k = 0; k < 3; ++k) normals[t[k]] += n;
    }
    for (auto& n : normals) {
        const float length = n.norm();
        if (length > 1e-8f) n /= length;
        else n.setZero();
    }
}

float ParallelCloth::maxRelativeError() const {
    float maximum = 0;
    for (const auto& c : links)
        maximum = std::max(maximum,
            std::abs((particles[c.b].pos - particles[c.a].pos).norm() - c.rest) / c.rest);
    return maximum;
}
