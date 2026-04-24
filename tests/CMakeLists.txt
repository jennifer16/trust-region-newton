# 单元测试1
add_executable(test_mesh_ops
    unit_tests/test_mesh_ops.cpp
)

#include directories
include_directories(/usr/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/include)

target_link_libraries(test_mesh_ops
    GTest::gtest
    GTest::gtest_main
)

# ===========================================

# 单元测试2
add_executable(test_cubic_regular
    unit_tests/test_cubic_regular.cpp
)

#include directories
include_directories(/usr/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/include)

target_link_libraries(test_cubic_regular
    GTest::gtest
    GTest::gtest_main
)

# add_executable(test_fsfem_data
#     unit_tests/test_fsfem_data.cpp
# )

# target_link_libraries(test_fsfem_data
#     fsfem_core
#     GTest::gtest
#     GTest::gtest_main
# )

# ==============================================
# 注册所有测试
include(GoogleTest)
gtest_discover_tests(test_mesh_ops)
gtest_discover_tests(test_cubic_regular)
# gtest_discover_tests(test_fsfem_data)