cmake_minimum_required(VERSION 3.20)
project(boostedpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++20 as required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Configure compiler options
if(MSVC)
    add_compile_options(/W4 /permissive- /EHsc /MP)
else()
    add_compile_options(-Wall -Wextra -pedantic)

    # Check if AVX2 is available
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        add_compile_options(-mavx2)
    endif()

    # Check if SSE4.2 is available
    check_cxx_compiler_flag(-msse4.2 COMPILER_SUPPORTS_SSE4_2)
    if(COMPILER_SUPPORTS_SSE4_2)
        add_compile_options(-msse4.2)
    endif()
endif()

# Find or fetch nlohmann_json
find_package(nlohmann_json 3.9.0 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# Define source files
set(BOOSTEDPP_HEADERS
    include/boostedpp/boostedpp.hpp
    include/boostedpp/data.hpp
    include/boostedpp/tree.hpp
    include/boostedpp/gbdt.hpp
    include/boostedpp/config.hpp
    include/boostedpp/metrics.hpp
    include/boostedpp/simd_utils.hpp
    include/boostedpp/serialization.hpp
)

set(BOOSTEDPP_SOURCES
    src/data.cpp
    src/tree.cpp
    src/gbdt.cpp
    src/metrics.cpp
    src/simd_utils.cpp
    src/serialization.cpp
    src/config.cpp
)

# Build the library
add_library(boostedpp ${BOOSTEDPP_SOURCES} ${BOOSTEDPP_HEADERS})
target_include_directories(boostedpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(boostedpp PUBLIC OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)

# Add CLI executable
add_executable(boostedpp_cli
    cli/main.cpp
    cli/train.cpp
    cli/predict.cpp
    cli/cv.cpp
)
set_target_properties(boostedpp_cli PROPERTIES OUTPUT_NAME boostedpp)
target_link_libraries(boostedpp_cli PRIVATE boostedpp)

# Installation
include(GNUInstallDirs)
install(TARGETS boostedpp boostedpp_cli
    EXPORT boostedpp-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT boostedpp-targets
    FILE boostedpp-targets.cmake
    NAMESPACE boostedpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/boostedpp
)

# Package configuration
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/boostedpp-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/boostedpp-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/boostedpp
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/boostedpp-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/boostedpp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/boostedpp-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/boostedpp
)

# Enable testing
option(BUILD_TESTING "Build the testing tree" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples
option(BUILD_EXAMPLES "Build the examples" ON)
if(BUILD_EXAMPLES)
    add_executable(simple_example examples/simple_example.cpp)
    target_link_libraries(simple_example PRIVATE boostedpp)
endif()

# Add documentation if Doxygen is found
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()
