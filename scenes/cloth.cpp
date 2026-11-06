#include "core/xpbd_gpu.hpp"

#include "raylib.h"
#include "rcamera.h"
#include "imgui.h"
#include "rlImGui.h"

#include <deque>
#include <vector>
#include <map>
#include <random>

struct Grab {
    int particle = -1;
    double distance = 0;
    double invMass = 0;
};

struct ClothGeometry {
    std::vector<Eigen::Vector3i> triangles;
    size_t stretchCount = 0;
};

xpbd::Solver makeCloth(int rows, int cols, double spacing, double particleMass, ClothGeometry& geometry) {
    xpbd::Solver state;
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
                geometry.triangles.push_back({a, c, d});
                geometry.triangles.push_back({a, d, b});
            } else {
                geometry.triangles.push_back({a, c, b});
                geometry.triangles.push_back({b, c, d});
            }
        }
    }

    auto makeConstraint = [&](int i, int j, double compliance = 0) {
        const double restLength = (state.particles[j].pos - state.particles[i].pos).norm();
        return xpbd::DistanceConstraint {i, j, restLength, compliance};
    };
    // Each edge stores the vertex opposite it in its first triangle.
    std::map<std::pair<int, int>, int> opposite;
    for (const auto& triangle : geometry.triangles) {
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
    geometry.stretchCount = state.constraints.size();
    state.constraints.insert(state.constraints.end(), bending.begin(), bending.end());
    return state;
}

Vector3 renderVector(const Eigen::Vector3d& p) {
    return {float(p.x()), float(p.y()), float(p.z())};
}

void complianceSlider(const char* label, std::vector<xpbd::DistanceConstraint>& constraints, size_t first, size_t last, double maximum) {
    if (first == last) return;
    double value = constraints[first].compliance;
    const double zero = 0;
    if (ImGui::SliderScalar(label, ImGuiDataType_Double, &value, &zero, &maximum, "%.3g",
        ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
        for (size_t i = first; i < last; ++i) constraints[i].compliance = value;
    }
}

void drawControls(xpbd::Solver& simulation, const ClothGeometry& geometry, bool& paused, bool& step, bool& reset) {
    ImGui::SetNextWindowPos({40, 40}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Cloth", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
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
        complianceSlider("Stretch compliance", simulation.constraints, 0, geometry.stretchCount, 1e-4);
        complianceSlider("Bending compliance", simulation.constraints, geometry.stretchCount, simulation.constraints.size(), 10);
        ImGui::PopItemWidth();

        ImGui::TextUnformatted("Space: pause | N: step | R: reset\n"
            "Left drag: grab | Right drag: orbit | Wheel: zoom");
    }

    ImGui::End();
}

void handleInput(xpbd::Solver& simulation, size_t stretchCount, bool& paused, bool& step, bool& reset) {
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
            for (size_t j = 0; j < stretchCount; ++j) simulation.constraints[j].compliance = values[i];
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

void drawCloth(const xpbd::Solver& simulation, const ClothGeometry& geometry) {
    DrawGrid(12, 1);
    for (const auto& t : geometry.triangles) {
        const Vector3 a = renderVector(simulation.particles[t[0]].pos);
        const Vector3 b = renderVector(simulation.particles[t[1]].pos);
        const Vector3 c = renderVector(simulation.particles[t[2]].pos);
        DrawTriangle3D(a, b, c, {48, 104, 148, 255});
        DrawTriangle3D(c, b, a, {48, 104, 148, 255});
    }
    for (size_t i = 0; i < geometry.stretchCount; ++i) {
        const auto& c = simulation.constraints[i];
        const auto& a = simulation.particles[c.i].pos;
        const auto& b = simulation.particles[c.j].pos;
        DrawLine3D(renderVector(a), renderVector(b), SKYBLUE);
    }
    for (const auto& p : simulation.particles) {
        DrawSphereEx(renderVector(p.pos), p.invMass == 0 ? 0.06f : 0.015f,
            8, 12, p.invMass == 0 ? ORANGE : BLUE);
    }
}

int main() {
    // Initialization
    ClothGeometry geometry;
    xpbd::CudaSolver simulation {makeCloth(17, 21, 0.18, 0.01, geometry)};
    const std::vector<xpbd::Particle> initialParticles = simulation.particles;
    constexpr double dt = 1.0 / 60;

    InitWindow(1000, 700, "dynamics | cloth");
    SetTargetFPS(60);

    rlImGuiSetup(true);
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().Fonts->Clear();
    ImGui::GetIO().Fonts->AddFontDefaultVector();
    ImGui::GetStyle().ScaleAllSizes(2);
    ImGui::GetStyle().FontScaleMain = 2;

    Camera3D camera {
        {5, 4.5f, 8},
        {0, 3, 0},
        {0, 1, 0},
        45,
        CAMERA_PERSPECTIVE
    };

    Grab grab;
    bool paused = false;

    while (!WindowShouldClose()) {
        BeginDrawing();
        rlImGuiBegin();

        // Controls
        bool step = false;
        bool reset = false;

        drawControls(simulation, geometry, paused, step, reset);
        handleInput(simulation, geometry.stretchCount, paused, step, reset);

        // Update
        if (reset) {
            simulation.particles = initialParticles;
            grab = {};
        }

        updateCamera(camera, grab.particle >= 0);
        updateGrab(simulation, camera, grab, 0.06f);

        if (!paused || step) {
            simulation.step(dt);
        }

        // Draw
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);
        drawCloth(simulation, geometry);
        EndMode3D();

        rlImGuiEnd();
        EndDrawing();
    }

    // Cleanup
    rlImGuiShutdown();
    CloseWindow();

    return 0;
}
