#include "core/mpm_gpu.hpp"

#include "raylib.h"
#include "imgui.h"
#include "rlImGui.h"

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

void drawControls(MpmSimulation& simulation, bool& paused, bool& step, bool& reset, bool& showGrid) {
    ImGui::SetNextWindowPos({32, 32}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("MLS-MPM", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;

        ImGui::SameLine();
        ImGui::BeginDisabled(!paused);
        step = ImGui::Button("Step");
        ImGui::EndDisabled();

        ImGui::SameLine();
        reset = ImGui::Button("Reset");

        ImGui::PushItemWidth(260);
        const double minE = 50, maxE = 2000, minNu = 0, maxNu = 0.45;
        ImGui::SliderScalar("Young's modulus", ImGuiDataType_Double, &simulation.young,
            &minE, &maxE, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderScalar("Poisson ratio", ImGuiDataType_Double, &simulation.poisson,
            &minNu, &maxNu, "%.2f");
        ImGui::PopItemWidth();
        ImGui::Checkbox("Show grid", &showGrid);
        ImGui::Text("%zu particles | %d x %d cells", simulation.particles.size(),
            simulation.resolution, simulation.resolution);
        ImGui::Text("Time %.2f s | %d substeps/frame",
            simulation.time, simulation.lastSubsteps);
        ImGui::TextUnformatted("CUDA fixed-corotated elastic solids\n"
            "APIC transfers | frictionless walls\nSpace: pause   N: step   R: reset");
    }

    ImGui::End();
}

void handleInput(bool& paused, bool& step, bool& reset) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (IsKeyPressed(KEY_SPACE)) paused = !paused;
    if (IsKeyPressed(KEY_N)) step = true;
    if (IsKeyPressed(KEY_R)) reset = true;
}

void drawMpm(const MpmSimulation& simulation, bool showGrid) {
    const float side = float(std::min(GetScreenWidth() - 400, GetScreenHeight() - 40));
    const float x = float(GetScreenWidth()) - side - 20, y = 20;
    if (showGrid) {
        for (int i = 0; i <= simulation.resolution; ++i) {
            const float offset = side * float(i) / simulation.resolution;
            DrawLineV({x + offset, y}, {x + offset, y + side}, {36, 42, 52, 255});
            DrawLineV({x, y + offset}, {x + side, y + offset}, {36, 42, 52, 255});
        }
    }
    const float margin = 2 * side / simulation.resolution;
    DrawRectangleLinesEx({x + margin, y + margin, side - 2 * margin, side - 2 * margin},
        2, GRAY);
    for (const auto& p : simulation.particles)
        DrawCircleV({x + side * float(p.pos.x()), y + side * float(1 - p.pos.y())},
            2.5f, p.group ? ORANGE : SKYBLUE);
}

int main() {
    // Initialization
    CudaMpmSimulation simulation {makeMpmScene()};
    constexpr double dt = 1.0 / 60;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1400, 900, "dynamics | 2D MLS-MPM");
    SetWindowMinSize(960, 600);
    SetTargetFPS(60);

    rlImGuiSetup(true);
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().Fonts->Clear();
    ImGui::GetIO().Fonts->AddFontDefaultVector();
    ImGui::GetStyle().ScaleAllSizes(2);
    ImGui::GetStyle().FontScaleMain = 2;

    bool paused = false;
    bool showGrid = false;

    while (!WindowShouldClose()) {
        BeginDrawing();
        rlImGuiBegin();

        // Controls
        bool step = false;
        bool reset = false;

        drawControls(simulation, paused, step, reset, showGrid);
        handleInput(paused, step, reset);

        // Update
        if (reset) simulation = CudaMpmSimulation {makeMpmScene()};
        if (!paused || step) simulation.step(dt);

        // Draw
        ClearBackground({20, 24, 32, 255});
        drawMpm(simulation, showGrid);

        rlImGuiEnd();
        EndDrawing();
    }

    // Cleanup
    rlImGuiShutdown();
    CloseWindow();

    return 0;
}
