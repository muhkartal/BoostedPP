/**
 * @file test_main.cpp
 * @brief Main entry point for tests.
 */

 #include <gtest/gtest.h>

 /**
  * @brief Main entry point for tests.
  *
  * @param argc Number of command-line arguments.
  * @param argv Command-line arguments.
  * @return Exit code.
  */
 int main(int argc, char** argv) {
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
 }
