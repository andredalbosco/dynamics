#include "xpbd.hpp"

#include "imgui.h"
#include "raylib.h"
#include "rlImGui.h"

#include <algorithm>

namespace {

constexpr int columns = 25;
constexpr int rows = 16;
constexpr double spacing = 22.0;

Vector2 toRaylib(const xpbd::Vector2& value)
{
    return {static_cast<float>(value.x()), static_cast<float>(value.y())};
}

xpbd::ParticleIndex index(int column, int row)
{
    return static_cast<xpbd::ParticleIndex>(row * columns + column);
}

xpbd::Solver makeCloth(double compliance)
{
    xpbd::Solver solver;
    const xpbd::Vector2 origin{270.0, 70.0};

    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            const bool pinned = row == 0 && (column == 0 || column == columns - 1);
            const xpbd::Vector2 position =
                origin + xpbd::Vector2{column * spacing, row * spacing};
            solver.addParticle(position, pinned ? 0.0 : 1.0);
        }
    }

    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            if (column + 1 < columns)
            {
                solver.addDistanceConstraint(
                    index(column, row), index(column + 1, row), compliance
                );
            }
            if (row + 1 < rows)
            {
                solver.addDistanceConstraint(
                    index(column, row), index(column, row + 1), compliance
                );
            }
            if (column + 1 < columns && row + 1 < rows)
            {
                solver.addDistanceConstraint(
                    index(column, row), index(column + 1, row + 1), compliance
                );
                solver.addDistanceConstraint(
                    index(column + 1, row), index(column, row + 1), compliance
                );
            }
        }
    }

    return solver;
}

void drawCloth(const xpbd::Solver& solver)
{
    const auto& particles = solver.particles();
    for (const xpbd::DistanceConstraint& constraint : solver.constraints())
    {
        DrawLineEx(
            toRaylib(particles[constraint.first].position),
            toRaylib(particles[constraint.second].position),
            1.0f,
            Color{105, 145, 190, 180}
        );
    }

    for (const xpbd::Particle& particle : particles)
    {
        DrawCircleV(
            toRaylib(particle.position),
            particle.isFixed() ? 5.0f : 2.0f,
            particle.isFixed() ? GOLD : RAYWHITE
        );
    }
}

} // namespace

int main()
{
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1100, 720, "XPBD cloth");
    SetTargetFPS(60);
    rlImGuiSetup(true);

    float compliance = 1.0e-6f;
    int substeps = 4;
    int iterations = 8;
    bool paused = false;
    xpbd::Solver solver = makeCloth(compliance);

    while (!WindowShouldClose())
    {
        if (!paused)
        {
            const double timeStep =
                std::clamp<double>(GetFrameTime(), 1.0 / 240.0, 1.0 / 30.0);
            for (xpbd::DistanceConstraint& constraint : solver.constraints())
            {
                constraint.compliance = compliance;
            }
            solver.step(timeStep, substeps, iterations, {0.0, 980.0});
        }

        BeginDrawing();
        ClearBackground(Color{24, 26, 32, 255});
        drawCloth(solver);

        rlImGuiBegin();
        ImGui::SetNextWindowSize({280.0f, 0.0f}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Cloth");
        ImGui::Checkbox("Paused", &paused);
        ImGui::SliderInt("Substeps", &substeps, 1, 16);
        ImGui::SliderInt("Iterations", &iterations, 1, 32);
        ImGui::SliderFloat(
            "Compliance", &compliance, 1.0e-8f, 1.0e-2f, "%.2e",
            ImGuiSliderFlags_Logarithmic
        );
        ImGui::Text("Particles: %zu", solver.particles().size());
        ImGui::Text("Constraints: %zu", solver.constraints().size());
        ImGui::Text("Max |C|: %.4f", solver.maximumConstraintError());
        if (ImGui::Button("Reset"))
        {
            solver = makeCloth(compliance);
        }
        ImGui::End();
        rlImGuiEnd();

        EndDrawing();
    }

    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
