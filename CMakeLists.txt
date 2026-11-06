cmake_minimum_required(VERSION 3.20)

if(APPLE)
    project(dynamics LANGUAGES C CXX OBJCXX)
else()
    project(dynamics LANGUAGES C CXX)
endif()

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

function(add_scene name)
    add_executable(${name} src/scenes/${name}.cpp)
    target_link_libraries(${name} PRIVATE xpbd rlimgui)
endfunction()

add_scene(cloth)
add_scene(pendulum)

enable_testing()
add_executable(xpbd_tests tests/xpbd.cpp)
target_link_libraries(xpbd_tests PRIVATE xpbd)
add_test(NAME xpbd_tests COMMAND xpbd_tests)

if(APPLE)
    add_subdirectory(experiments/metal_vec_sum)
endif()
