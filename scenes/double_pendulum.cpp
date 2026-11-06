#include "core/xpbd_gpu.hpp"

#include "raylib.h"
#include "rcamera.h"
#include "imgui.h"
#include "rlImGui.h"

#include <deque>
#include <vector>

struct Grab {
    int particle = -1;
    double distance = 0;
    double invMass = 0;
};

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

Vector3 renderVector(const Eigen::Vector3d& p) {
    return {float(p.x()), float(p.y()), float(p.z())};
}

void complianceSlider(const char* label, std::vector<xpbd::DistanceConstraint>& constraints, double maximum) {
    if (constraints.empty()) return;
    double value = constraints.front().compliance;
    const double zero = 0;
    if (ImGui::SliderScalar(label, ImGuiDataType_Double, &value, &zero, &maximum, "%.3g",
        ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
        for (auto& c : constraints) c.compliance = value;
    }
}

void drawControls(xpbd::Solver& simulation, bool& paused, bool& step, bool& reset) {
    ImGui::SetNextWindowPos({40, 40}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Double pendulum", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("CUDA XPBD");

        if (ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;

        ImGui::SameLine();
        ImGui::BeginDisabled(!paused);
        step = ImGui::Button("Step");
        ImGui::EndDisabled();

        ImGui::SameLine();
        reset = ImGui::Button("Reset");

        ImGui::PushItemWidth(360);
        ImGui::SliderInt("Substeps", &simulation.substeps, 1, 256,
            "%d", ImGuiSliderFlags_AlwaysClamp);
        complianceSlider("Stretch compliance", simulation.constraints, 1);
        ImGui::PopItemWidth();

        ImGui::TextUnformatted("Space: pause | N: step | R: reset\n"
            "Left drag: grab | Right drag: orbit | Wheel: zoom");
    }

    ImGui::End();
}

void handleInput(xpbd::Solver& simulation, bool& paused, bool& step, bool& reset) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (IsKeyPressed(KEY_SPACE)) paused = !paused;
    if (IsKeyPressed(KEY_R)) reset = true;
    if (IsKeyPressed(KEY_N)) step = true;
    const int change = int(IsKeyPressed(KEY_RIGHT)) - int(IsKeyPressed(KEY_LEFT));
    if (change) {
        simulation.substeps = std::clamp(simulation.substeps + change, 1, 256);
    }
    const int keys[] {KEY_ONE, KEY_TWO, KEY_THREE};
    const double values[] {0, 1e-6, 1e-4};
    for (int i = 0; i < 3; ++i) {
        if (IsKeyPressed(keys[i])) {
            for (auto& c : simulation.constraints) c.compliance = values[i];
        }
    }
}

void updateCamera(Camera3D& camera, bool dragging) {
    if (dragging || ImGui::GetIO().WantCaptureMouse) return;

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        const Vector2 delta = GetMouseDelta();
        CameraYaw(&camera, -delta.x * 0.005f, true);
        CameraPitch(&camera, -delta.y * 0.005f, true, true, false);
    }
    const float wheel = GetMouseWheelMove();
    if (camera.projection == CAMERA_ORTHOGRAPHIC) {
        camera.fovy = std::clamp(camera.fovy * std::pow(0.9f, wheel), 1.0f, 30.0f);
    } else {
        CameraMoveToTarget(&camera, -wheel * 0.5f);
    }
}

void updateGrab(xpbd::Solver& simulation, const Camera3D& camera, Grab& grab, float radius) {
    const Ray ray = GetScreenToWorldRay(GetMousePosition(), camera);
    const Eigen::Vector3d origin {ray.position.x, ray.position.y, ray.position.z};
    const Eigen::Vector3d direction {ray.direction.x, ray.direction.y, ray.direction.z};
    if (grab.particle < 0 && !ImGui::GetIO().WantCaptureMouse && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        double nearest = 1e30;
        for (int i = 0; i < int(simulation.particles.size()); ++i) {
            const auto& p = simulation.particles[i];
            if (p.invMass == 0) continue;
            const auto hit = GetRayCollisionSphere(ray, renderVector(p.pos), radius);
            if (hit.hit && hit.distance < nearest) {
                nearest = hit.distance;
                grab.particle = i;
                grab.distance = (p.pos - origin).dot(direction);
                grab.invMass = p.invMass;
            }
        }
    }
    if (grab.particle < 0) return;

    auto& p = simulation.particles[grab.particle];
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        p.pos = origin + direction * grab.distance;
        p.prevPos = p.pos;
        p.invMass = 0;
    } else {
        p.invMass = grab.invMass;
        grab.particle = -1;
    }
}

void drawDoublePendulum(const xpbd::Solver& simulation, const std::deque<Eigen::Vector3d>& trail) {
    for (size_t i = 1; i < trail.size(); ++i) {
        DrawLine3D(renderVector(trail[i - 1]), renderVector(trail[i]), {68, 124, 140, 255});
    }
    for (const auto& c : simulation.constraints) {
        const auto& a = simulation.particles[c.i].pos;
        const auto& b = simulation.particles[c.j].pos;
        DrawCylinderEx(renderVector(a), renderVector(b), 0.018f, 0.018f, 8, SKYBLUE);
    }
    for (const auto& p : simulation.particles) {
        DrawSphereEx(renderVector(p.pos), p.invMass == 0 ? 0.06f : 0.1f,
            8, 12, p.invMass == 0 ? ORANGE : BLUE);
    }
}

int main() {
    // Initialization
    xpbd::CudaSolver simulation {makeDoublePendulum()};
    const std::vector<xpbd::Particle> initialParticles = simulation.particles;
    constexpr double dt = 1.0 / 60;

    InitWindow(1000, 700, "dynamics | double_pendulum");
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

    Grab grab;
    bool paused = false;
    std::deque<Eigen::Vector3d> trail;

    while (!WindowShouldClose()) {
        BeginDrawing();
        rlImGuiBegin();

        // Controls
        bool step = false;
        bool reset = false;

        drawControls(simulation, paused, step, reset);
        handleInput(simulation, paused, step, reset);

        // Update
        if (reset) {
            simulation.particles = initialParticles;
            grab = {};
            trail.clear();
        }

        updateCamera(camera, grab.particle >= 0);
        updateGrab(simulation, camera, grab, 0.13f);

        if (!paused || step) {
            simulation.step(dt);
            trail.push_back(simulation.particles.back().pos);
            if (trail.size() > 1000) trail.pop_front();
        }

        // Draw
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);
        drawDoublePendulum(simulation, trail);
        EndMode3D();

        rlImGuiEnd();
        EndDrawing();
    }

    // Cleanup
    rlImGuiShutdown();
    CloseWindow();

    return 0;
}
