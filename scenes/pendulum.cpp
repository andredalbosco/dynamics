#include "core/xpbd.hpp"
#include "core/vbd.hpp"
#include "core/avbd.hpp"

#include "raylib.h"
#include "imgui.h"
#include "rlImGui.h"

#include <vector>

void drawPendulum(const Eigen::Vector3d& anchor, const Eigen::Vector3d& bob) {
    const Vector3 a {float(anchor.x()), float(anchor.y()), float(anchor.z())};
    const Vector3 b {float(bob.x()), float(bob.y()), float(bob.z())};

    DrawCylinderEx(a, b, 0.018f, 0.018f, 8, DARKGRAY);
    DrawSphere(a, 0.06f, ORANGE);
    DrawSphere(b, 0.1f, BLUE);
}

void drawControls(bool& paused, bool& step, bool& reset, int& solver) {
    ImGui::SetNextWindowPos({40, 40}, ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Pendulum", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;

        ImGui::SameLine();
        ImGui::BeginDisabled(!paused);
        step = ImGui::Button("Step");
        ImGui::EndDisabled();

        ImGui::SameLine();
        reset = ImGui::Button("Reset");

        ImGui::Combo("Solver", &solver, "XPBD\0VBD\0AVBD\0");
        if (solver != 1) ImGui::TextUnformatted("Hard distance constraint");
        else ImGui::TextUnformatted("Elastic link: stiffness 1e6 N/m");
        ImGui::TextUnformatted("Space: pause   N: step   R: reset");
    }

    ImGui::End();
}

void handleInput(bool& paused, bool& step, bool& reset) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (IsKeyPressed(KEY_SPACE)) paused = !paused;
    if (IsKeyPressed(KEY_N)) step = true;
    if (IsKeyPressed(KEY_R)) reset = true;
}

int main() {
    // Initialization
    const Eigen::Vector3d anchor {0, 3, 0};
    const Eigen::Vector3d bob = anchor + Eigen::Vector3d(std::sin(0.65), -std::cos(0.65), 0);

    const std::vector<xpbd::Particle> initialParticles = {
        {anchor, anchor, Eigen::Vector3d::Zero(), 0},
        {bob, bob, Eigen::Vector3d::Zero(), 1}
    };

    xpbd::Solver pendulum;
    pendulum.particles = initialParticles;
    pendulum.constraints = {{0, 1, 1, 0}};

    vbd::Solver vbd;
    for (const auto& p : pendulum.particles)
        vbd.particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
    vbd.energies = {{0, 1, 1, 1e6}};
    vbd.substeps = pendulum.substeps;

    constexpr double dt = 1.0 / 60;
    pendulum.dt = dt;

    avbd::Solver avbd;
    for (const auto& p : initialParticles)
        avbd.particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
    avbd.constraints = {{0, 1, 1, 0}};
    avbd.substeps = pendulum.substeps;
    avbd.dt = dt;

    InitWindow(1000, 700, "dynamics | pendulum");
    SetTargetFPS(60);

    rlImGuiSetup(true);
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().Fonts->Clear();
    ImGui::GetIO().Fonts->AddFontDefaultVector();
    ImGui::GetStyle().ScaleAllSizes(2);
    ImGui::GetStyle().FontScaleMain = 2;

    Camera3D camera {
        {0, 2, 8},
        {0, 2, 0},
        {0, 1, 0},
        6,
        CAMERA_ORTHOGRAPHIC
    };

    bool paused = false;
    int solver = 0;

    while (!WindowShouldClose()) {
        BeginDrawing();
        rlImGuiBegin();

        // Controls
        bool step = false;
        bool reset = false;

        drawControls(paused, step, reset, solver);

        // Update
        handleInput(paused, step, reset);

        if (reset) {
            pendulum.particles = initialParticles;
            for (size_t i = 0; i < initialParticles.size(); ++i) {
                const auto& p = initialParticles[i];
                vbd.particles[i] = {p.pos, p.prevPos, p.vel, p.invMass};
                avbd.particles[i] = {p.pos, p.prevPos, p.vel, p.invMass};
            }
            avbd.constraints = {{0, 1, 1, 0}};
        }
        if (!paused || step) {
            pendulum.step();
            vbd.step();
            avbd.step();
        }

        // Draw
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);
        if (solver == 0) drawPendulum(pendulum.particles[0].pos, pendulum.particles[1].pos);
        else if (solver == 1) drawPendulum(vbd.particles[0].pos, vbd.particles[1].pos);
        else drawPendulum(avbd.particles[0].pos, avbd.particles[1].pos);
        EndMode3D();

        rlImGuiEnd();
        EndDrawing();
    }

    // Cleanup
    rlImGuiShutdown();
    CloseWindow();

    return 0;
}
