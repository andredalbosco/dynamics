#include "core/xpbd.hpp"
#include "core/vbd.hpp"
#include "core/pd.hpp"
#include "core/avbd.hpp"
#include "core/bw.hpp"

#include "raylib.h"
#include "imgui.h"
#include "rlImGui.h"

#include <vector>

void drawSpring(const Eigen::Vector3d& anchor, const Eigen::Vector3d& bob) {
    const Vector3 a {float(anchor.x()), float(anchor.y()), float(anchor.z())};
    const Vector3 b {float(bob.x()), float(bob.y()), float(bob.z())};

    DrawCylinderEx(a, b, 0.018f, 0.018f, 8, DARKGRAY);
    DrawSphere(a, 0.06f, ORANGE);
    DrawSphere(b, 0.1f, BLUE);
}

void drawControls(bool& paused, bool& step, bool& reset, int& solver) {
    ImGui::SetNextWindowPos({40, 40}, ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Spring", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;

        ImGui::SameLine();
        ImGui::BeginDisabled(!paused);
        step = ImGui::Button("Step");
        ImGui::EndDisabled();

        ImGui::SameLine();
        reset = ImGui::Button("Reset");

        ImGui::Combo("Solver", &solver, "XPBD\0VBD\0PD\0AVBD\0BW\0");

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
    const Eigen::Vector3d anchor {-1, 2.5, 0};
    const Eigen::Vector3d bob {0.5, 2.5, 0};

    const std::vector<xpbd::Particle> initialParticles = {
        {anchor, anchor, Eigen::Vector3d::Zero(), 0},
        {bob, bob, Eigen::Vector3d::Zero(), 1}
    };

    xpbd::Solver spring;
    spring.particles = initialParticles;
    constexpr double compliance = 0.01;
    spring.constraints = {{0, 1, 1, compliance}};
    spring.gravity.setZero();

    constexpr double dt = 1.0 / 60;
    spring.dt = dt;

    vbd::Solver vbd;
    for (const auto& p : initialParticles)
        vbd.particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
    vbd.energies = {{0, 1, 1, 1 / compliance}};
    vbd.gravity.setZero();
    vbd.substeps = spring.substeps;
    vbd.dt = dt;

    pd::Solver pd;
    auto& ps = pd.particles;
    ps.resize(2);
    ps.pos.row(0) = anchor.transpose();
    ps.pos.row(1) = bob.transpose();
    ps.prevPos = ps.pos;
    ps.invMass << 0, 1;
    pd.energies = {{0, 1, 1, 1 / compliance}};
    pd.gravity.setZero();
    pd.substeps = spring.substeps;
    pd.dt = dt;
    pd.factorize();

    avbd::Solver avbd;
    for (const auto& p : initialParticles)
        avbd.particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
    avbd.constraints = {{0, 1, 1, compliance}};
    avbd.gravity.setZero();
    avbd.substeps = spring.substeps;
    avbd.dt = dt;

    bw::Solver bw;
    for (const auto& p : initialParticles)
        bw.particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
    bw.energies = {{0, 1, 1, 1 / compliance}};
    bw.gravity.setZero();
    bw.substeps = spring.substeps;
    bw.dt = dt;

    InitWindow(1600, 1000, "dynamics | spring");
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
            spring.particles = initialParticles;
            ps.pos.row(0) = anchor.transpose();
            ps.pos.row(1) = bob.transpose();
            ps.prevPos = ps.pos;
            ps.inertialPos = ps.pos;
            ps.vel.setZero();
            for (size_t i = 0; i < initialParticles.size(); ++i) {
                const auto& p = initialParticles[i];
                vbd.particles[i] = {p.pos, p.prevPos, p.vel, p.invMass};
                avbd.particles[i] = {p.pos, p.prevPos, p.vel, p.invMass};
                bw.particles[i] = {p.pos, p.prevPos, p.vel, p.invMass};
            }
            avbd.constraints = {{0, 1, 1, compliance}};
        }
        if (!paused || step) {
            spring.step();
            vbd.step();
            pd.step();
            avbd.step();
            bw.step();
        }

        // Draw
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);
        if (solver == 0) drawSpring(spring.particles[0].pos, spring.particles[1].pos);
        else if (solver == 1) drawSpring(vbd.particles[0].pos, vbd.particles[1].pos);
        else if (solver == 2) drawSpring(ps.pos.row(0).transpose(), ps.pos.row(1).transpose());
        else if (solver == 3) drawSpring(avbd.particles[0].pos, avbd.particles[1].pos);
        else drawSpring(bw.particles[0].pos, bw.particles[1].pos);
        EndMode3D();

        rlImGuiEnd();
        EndDrawing();
    }

    // Cleanup
    rlImGuiShutdown();
    CloseWindow();

    return 0;
}
