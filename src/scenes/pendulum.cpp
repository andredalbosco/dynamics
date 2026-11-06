#include "xpbd.hpp"

#include "imgui.h"
#include "raylib.h"
#include "rlImGui.h"

#include <algorithm>

namespace {

constexpr int particleCount = 12;
constexpr double segmentLength = 36.0;

Vector2 toRaylib(const xpbd::Vector2& value)
{
    return {static_cast<float>(value.x()), static_cast<float>(value.y())};
}

xpbd::Solver makePendulum(double compliance)
{
    xpbd::Solver solver;
    const xpbd::Vector2 anchor{500.0, 90.0};
    const xpbd::Vector2 direction{0.65, 0.76};

    for (int particle = 0; particle < particleCount; ++particle)
    {
        const xpbd::Vector2 position =
            anchor + static_cast<double>(particle) * segmentLength * direction;
        solver.addParticle(position, particle == 0 ? 0.0 : 1.0);
        if (particle > 0)
        {
            solver.addDistanceConstraint(
                static_cast<xpbd::ParticleIndex>(particle - 1),
                static_cast<xpbd::ParticleIndex>(particle),
                segmentLength,
                compliance
            );
        }
    }

    return solver;
}

void drawPendulum(const xpbd::Solver& solver)
{
    const auto& particles = solver.particles();
    for (const xpbd::DistanceConstraint& constraint : solver.constraints())
    {
        DrawLineEx(
            toRaylib(particles[constraint.first].position),
            toRaylib(particles[constraint.second].position),
            4.0f,
            Color{105, 145, 190, 255}
        );
    }

    for (const xpbd::Particle& particle : particles)
    {
        DrawCircleV(
            toRaylib(particle.position),
            particle.isFixed() ? 8.0f : 7.0f,
            particle.isFixed() ? GOLD : RAYWHITE
        );
    }
}

} // namespace

int main()
{
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1000, 700, "XPBD pendulum");
    SetTargetFPS(60);
    rlImGuiSetup(true);

    float compliance = 1.0e-7f;
    int substeps = 4;
    int iterations = 8;
    bool paused = false;
    xpbd::Solver solver = makePendulum(compliance);

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
        drawPendulum(solver);

        rlImGuiBegin();
        ImGui::SetNextWindowSize({280.0f, 0.0f}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Pendulum");
        ImGui::Checkbox("Paused", &paused);
        ImGui::SliderInt("Substeps", &substeps, 1, 16);
        ImGui::SliderInt("Iterations", &iterations, 1, 32);
        ImGui::SliderFloat(
            "Compliance", &compliance, 1.0e-9f, 1.0e-2f, "%.2e",
            ImGuiSliderFlags_Logarithmic
        );
        ImGui::Text("Max |C|: %.4f", solver.maximumConstraintError());
        if (ImGui::Button("Reset"))
        {
            solver = makePendulum(compliance);
        }
        ImGui::End();
        rlImGuiEnd();

        EndDrawing();
    }

    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
