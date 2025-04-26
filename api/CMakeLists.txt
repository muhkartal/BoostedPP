cmake_minimum_required(VERSION 3.20)
project(boostedpp_api VERSION 0.1.0 LANGUAGES CXX)

# Set C++20 as required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the BoostedPP package
find_package(boostedpp REQUIRED)

# Find or fetch Crow
include(FetchContent)
FetchContent_Declare(
  crow
  GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
  GIT_TAG v1.0+5
)
FetchContent_MakeAvailable(crow)

# Build the API server
add_executable(boostedpp_api
  server.cpp
)

target_link_libraries(boostedpp_api PRIVATE
  boostedpp::boostedpp
  Crow::Crow
)

# Install
install(TARGETS boostedpp_api
  RUNTIME DESTINATION bin
)
