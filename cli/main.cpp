/**
 * @file main.cpp
 * @brief Main entry point for the BoostedPP CLI.
 */

 #include <iostream>
 #include <string>
 #include <memory>
 #include <unordered_map>
 #include <functional>

 #include "boostedpp/boostedpp.hpp"

 // Function prototypes for CLI subcommands
 int train_main(int argc, char** argv);
 int predict_main(int argc, char** argv);
 int cv_main(int argc, char** argv);

 /**
  * @brief Main entry point for the BoostedPP CLI.
  *
  * @param argc Number of command-line arguments.
  * @param argv Command-line arguments.
  * @return Exit code.
  */
 int main(int argc, char** argv) {
     if (argc < 2) {
         std::cerr << "Usage: boostedpp <command> [options]" << std::endl;
         std::cerr << "Commands:" << std::endl;
         std::cerr << "  train    Train a model" << std::endl;
         std::cerr << "  predict  Make predictions" << std::endl;
         std::cerr << "  cv       Cross-validation" << std::endl;
         return 1;
     }

     // Map of subcommands to their entry points
     std::unordered_map<std::string, std::function<int(int, char**)>> commands = {
         {"train", train_main},
         {"predict", predict_main},
         {"cv", cv_main}
     };

     // Check if the command exists
     std::string command = argv[1];
     auto it = commands.find(command);
     if (it == commands.end()) {
         std::cerr << "Unknown command: " << command << std::endl;
         return 1;
     }

     // Execute the command
     try {
         return it->second(argc - 1, argv + 1);
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 }
