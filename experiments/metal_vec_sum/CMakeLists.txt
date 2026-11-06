add_executable(metal_vec_sum
    main.cpp
    metal_vector_sum.mm
)

target_compile_definitions(metal_vec_sum PRIVATE
    METAL_VEC_SUM_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/vector_sum.metal"
)

target_link_libraries(metal_vec_sum PRIVATE
    "-framework Foundation"
    "-framework Metal"
)
