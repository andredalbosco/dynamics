#include <Eigen/Core>

#include "imgui.h"
#include "raylib.h"
#include "rlImGui.h"

int main()
{
    constexpr int screenWidth = 960;
    constexpr int screenHeight = 540;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(screenWidth, screenHeight, "dynamics");
    SetTargetFPS(60);
    rlImGuiSetup(true);

    Eigen::Vector2f textPosition{40.0f, 40.0f};

    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(Color{24, 26, 32, 255});

        DrawText("Hello, raylib!", static_cast<int>(textPosition.x()),
                 static_cast<int>(textPosition.y()), 32, RAYWHITE);

        rlImGuiBegin();
        ImGui::Begin("Dynamics");
        ImGui::TextUnformatted("Hello from Dear ImGui.");
        ImGui::SliderFloat2("Raylib text position", textPosition.data(), 0.0f,
                            static_cast<float>(GetScreenWidth()));
        ImGui::End();
        rlImGuiEnd();

        EndDrawing();
    }

    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
