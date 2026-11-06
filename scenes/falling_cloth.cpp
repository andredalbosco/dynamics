#include "core/parallel_cloth_gpu.hpp"

#include "raylib.h"
#include "rcamera.h"
#include "rlgl.h"
#include "imgui.h"
#include "rlImGui.h"

#include <algorithm>

Vector3 renderVector(const ParallelCloth::Vec& p) { return {p.x(), p.y(), p.z()}; }

// Expanded triangles avoid raylib's 16-bit index limit on large cloth meshes.
struct ClothMesh {
    Mesh mesh {};
    Material material {};
};

void fillClothMesh(Mesh& mesh, const ParallelCloth& cloth) {
    int vertex = 0;
    for (const auto& t : cloth.triangles)
        for (int k = 0; k < 3; ++k, ++vertex)
            for (int axis = 0; axis < 3; ++axis) {
                mesh.vertices[3 * vertex + axis] = cloth.particles[t[k]].pos[axis];
                mesh.normals[3 * vertex + axis] = cloth.normals[t[k]][axis];
            }
}

ClothMesh loadClothMesh(const ParallelCloth& cloth) {
    Mesh mesh {};
    mesh.vertexCount = int(cloth.triangles.size()) * 3;
    mesh.triangleCount = int(cloth.triangles.size());
    mesh.vertices = static_cast<float*>(MemAlloc(mesh.vertexCount * 3 * sizeof(float)));
    mesh.normals = static_cast<float*>(MemAlloc(mesh.vertexCount * 3 * sizeof(float)));
    fillClothMesh(mesh, cloth);
    UploadMesh(&mesh, true);
    Material material = LoadMaterialDefault();
    material.shader = LoadShaderFromMemory(
        "#version 330\n"
        "in vec3 vertexPosition; in vec3 vertexNormal; uniform mat4 mvp;\n"
        "out vec3 normal; void main() { normal = vertexNormal;"
        "gl_Position = mvp * vec4(vertexPosition, 1.0); }",
        "#version 330\n"
        "in vec3 normal; out vec4 finalColor; void main() {"
        "float light = 0.3 + 0.7 * abs(dot(normalize(normal), normalize(vec3(1,2,3))));"
        "finalColor = vec4(vec3(0.18,0.5,0.8) * light, 1.0); }");
    return {mesh, material};
}

void updateClothMesh(ClothMesh& mesh, const ParallelCloth& cloth) {
    fillClothMesh(mesh.mesh, cloth);
    UpdateMeshBuffer(mesh.mesh, 0, mesh.mesh.vertices, mesh.mesh.vertexCount * 3 * int(sizeof(float)), 0);
    UpdateMeshBuffer(mesh.mesh, 2, mesh.mesh.normals, mesh.mesh.vertexCount * 3 * int(sizeof(float)), 0);
}

void drawControls(ParallelCloth& cloth, bool& paused, bool& step, bool& reset,
    bool& hidden, bool& wireframe) {
    const char* modes[] = {"Sequential", "Coloring / hybrid", "Jacobi scatter", "Jacobi gather"};
    int mode = static_cast<int>(cloth.mode);

    ImGui::SetNextWindowPos({40, 40}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Falling cloth", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::PushItemWidth(400);
        if (ImGui::Combo("Solver", &mode, modes, IM_ARRAYSIZE(modes))) {
            cloth.mode = static_cast<ParallelCloth::Mode>(mode);
            reset = true;
        }
        if (ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;
        ImGui::SameLine();
        ImGui::BeginDisabled(!paused);
        step = ImGui::Button("Step");
        ImGui::EndDisabled();
        ImGui::SameLine();
        reset |= ImGui::Button("Reset");
        ImGui::SliderInt("Substeps", &cloth.substeps, 1, 64, "%d", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Jacobi relaxation", &cloth.jacobiScale, 0.01f, 0.35f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Friction", &cloth.friction, 0, 1, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::Checkbox("Sphere contact", &cloth.sphereEnabled);
        ImGui::Checkbox("Ground contact", &cloth.groundEnabled);
        ImGui::Checkbox("Hide cloth", &hidden);
        ImGui::Checkbox("Wireframe", &wireframe);
        ImGui::Text("%zu particles | %zu constraints", cloth.particles.size(), cloth.links.size());
        ImGui::TextUnformatted("CUDA | hard distance constraints\n"
            "Left drag: grab | Right drag: orbit | Middle: pan\n"
            "Wheel: zoom | WASD/QE: move\nSpace/P: pause | N: step | R: reset | H: hide\n"
            "C: hybrid | J: Jacobi | G: gather");
        ImGui::PopItemWidth();
    }
    ImGui::End();
}

void handleInput(ParallelCloth& cloth, bool& paused, bool& step, bool& reset, bool& hidden) {
    int mode = static_cast<int>(cloth.mode);
    const bool keyboard = !ImGui::GetIO().WantCaptureKeyboard;
    if (keyboard) {
        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_P)) paused = !paused;
        step |= paused && IsKeyPressed(KEY_N);
        reset |= IsKeyPressed(KEY_R);
        if (IsKeyPressed(KEY_H)) hidden = !hidden;
        int selected = mode;
        if (IsKeyPressed(KEY_C)) selected = 1;
        if (IsKeyPressed(KEY_J)) selected = 2;
        if (IsKeyPressed(KEY_G)) selected = 3;
        if (selected != mode) {
            mode = selected;
            cloth.mode = static_cast<ParallelCloth::Mode>(mode);
            reset = true;
        }
    }
}

void updateCamera(Camera3D& camera, int grabbed) {
    if (grabbed < 0 && !ImGui::GetIO().WantCaptureMouse) {
        const Vector2 delta = GetMouseDelta();
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            CameraYaw(&camera, -delta.x * 0.005f, true);
            CameraPitch(&camera, -delta.y * 0.005f, true, true, false);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            CameraMoveRight(&camera, -delta.x * 0.005f, false);
            CameraMoveUp(&camera, delta.y * 0.005f);
        }
        CameraMoveToTarget(&camera, -GetMouseWheelMove() * 0.3f);
        if (!ImGui::GetIO().WantCaptureKeyboard) {
            const float movement = std::min(GetFrameTime(), 0.1f) * 2;
            CameraMoveForward(&camera, movement * (IsKeyDown(KEY_W) - IsKeyDown(KEY_S)), false);
            CameraMoveRight(&camera, movement * (IsKeyDown(KEY_D) - IsKeyDown(KEY_A)), false);
            CameraMoveUp(&camera, movement * (IsKeyDown(KEY_E) - IsKeyDown(KEY_Q)));
        }
    }
}

struct Grab {
    int particle = -1;
    float distance = 0;
    float invMass = 0;
};

void updateGrab(ParallelCloth& cloth, const Camera3D& camera, Grab& grab, bool hidden) {
    const Ray ray = GetScreenToWorldRay(GetMousePosition(), camera);
    if (!hidden && grab.particle < 0 && !ImGui::GetIO().WantCaptureMouse && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        float closest = 1e30f;
        for (int i = 0; i < int(cloth.particles.size()); ++i) {
            const auto hit = GetRayCollisionSphere(ray, renderVector(cloth.particles[i].pos), 0.025f);
            if (hit.hit && hit.distance < closest && cloth.particles[i].invMass > 0) {
                closest = hit.distance; grab.particle = i;
            }
        }
        if (grab.particle >= 0) {
            grab.invMass = cloth.particles[grab.particle].invMass;
            const auto p = renderVector(cloth.particles[grab.particle].pos);
            grab.distance = (p.x-ray.position.x)*ray.direction.x +
                (p.y-ray.position.y)*ray.direction.y + (p.z-ray.position.z)*ray.direction.z;
            cloth.particles[grab.particle].invMass = 0;
        }
    }
    if (grab.particle >= 0) {
        auto& p = cloth.particles[grab.particle];
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            const Vector3 target {ray.position.x+ray.direction.x*grab.distance,
                ray.position.y+ray.direction.y*grab.distance, ray.position.z+ray.direction.z*grab.distance};
            p.pos = {target.x, target.y, target.z};
        } else { p.invMass = grab.invMass; grab.particle = -1; }
    }
}

void drawCloth(const ParallelCloth& cloth, const ClothMesh& mesh, bool hidden, bool wireframe) {
    DrawGrid(12, 1);
    if (cloth.sphereEnabled) DrawSphere(renderVector(cloth.sphereCenter), cloth.sphereRadius, GRAY);
    if (!hidden) {
        rlDisableBackfaceCulling();
        if (wireframe) rlEnableWireMode();
        DrawMesh(mesh.mesh, mesh.material, Matrix {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1});
        if (wireframe) rlDisableWireMode();
        rlEnableBackfaceCulling();
    }
}

int main() {
    // Initialization
    CudaParallelCloth cloth(140, 140);
    constexpr float dt = 1.0f / 60;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1600, 1000, "dynamics | Falling cloth");
    SetWindowMinSize(1000, 640);
    SetTargetFPS(60);

    rlImGuiSetup(true);
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().Fonts->Clear();
    ImGui::GetIO().Fonts->AddFontDefaultVector();
    ImGui::GetStyle().ScaleAllSizes(2);
    ImGui::GetStyle().FontScaleMain = 2;

    Camera3D camera {{3, 3.5f, 4}, {0, 1.4f, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};
    Grab grab;
    bool paused = false;
    bool hidden = false;
    bool wireframe = false;

    ClothMesh mesh = loadClothMesh(cloth);

    while (!WindowShouldClose()) {
        BeginDrawing();
        rlImGuiBegin();

        // Controls
        bool step = false;
        bool reset = false;

        drawControls(cloth, paused, step, reset, hidden, wireframe);
        handleInput(cloth, paused, step, reset, hidden);

        // Update
        if (reset) {
            cloth.reset();
            grab = {};
        }

        updateCamera(camera, grab.particle);
        updateGrab(cloth, camera, grab, hidden);

        if (!paused || step) cloth.step(dt);

        cloth.updateNormals();
        updateClothMesh(mesh, cloth);

        // Draw
        ClearBackground({20, 24, 32, 255});

        BeginMode3D(camera);
        drawCloth(cloth, mesh, hidden, wireframe);
        EndMode3D();

        rlImGuiEnd();
        EndDrawing();
    }

    // Cleanup
    UnloadMesh(mesh.mesh);
    UnloadMaterial(mesh.material);

    rlImGuiShutdown();
    CloseWindow();

    return 0;
}
