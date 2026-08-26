cmake_minimum_required(VERSION 3.20)

project(dynamics LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)

add_subdirectory(external/raylib EXCLUDE_FROM_ALL)
add_subdirectory(external/eigen EXCLUDE_FROM_ALL)

add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
)
target_include_directories(imgui PUBLIC external/imgui)

add_library(rlimgui STATIC external/rlimgui/rlImGui.cpp)
target_include_directories(rlimgui PUBLIC external/rlimgui)
target_link_libraries(rlimgui PUBLIC imgui raylib)

add_library(xpbd STATIC src/core/xpbd.cpp)
target_include_directories(xpbd PUBLIC src/core)
target_link_libraries(xpbd PUBLIC Eigen3::Eigen)

add_executable(dynamics
    src/main.cpp
    src/scenes/cloth.cpp
    src/scenes/pendulum.cpp
)
target_link_libraries(dynamics PRIVATE xpbd rlimgui)
