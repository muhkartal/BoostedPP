find_package(GTest)

if(GTest_FOUND)
    enable_testing()

    set(TEST_SOURCES
        test_main.cpp
        test_data.cpp
        test_tree.cpp
        test_gbdt.cpp
        test_simd_utils.cpp
    )

    add_executable(boostedpp_test ${TEST_SOURCES})
    target_link_libraries(boostedpp_test PRIVATE boostedpp GTest::GTest GTest::Main)

    add_test(NAME boostedpp_test COMMAND boostedpp_test)
else()
    message(STATUS "Google Test not found, skipping tests")
endif()
