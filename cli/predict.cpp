/**
 * @file predict.cpp
 * @brief Implementation of the predict CLI subcommand.
 */

 #include <iostream>
 #include <string>
 #include <unordered_map>
 #include <cstdlib>
 #include <fstream>

 #include "boostedpp/boostedpp.hpp"

 /**
  * @brief Print usage information for the predict subcommand.
  */
 void print_predict_usage() {
     std::cerr << "Usage: boostedpp predict [options]" << std::endl;
     std::cerr << "Options:" << std::endl;
     std::cerr << "  --data <file>       Input data file (CSV format)" << std::endl;
     std::cerr << "  --model <file>      Model file path" << std::endl;
     std::cerr << "  --out <file>        Output prediction file path" << std::endl;
     std::cerr << "  --nthreads <int>    Number of threads (-1 = all) (default: -1)" << std::endl;
 }

 /**
  * @brief Parse command line arguments for the predict subcommand.
  *
  * @param argc Number of command-line arguments.
  * @param argv Command-line arguments.
  * @param data_file Output parameter for data file path.
  * @param model_file Output parameter for model file path.
  * @param output_file Output parameter for output file path.
  * @param n_threads Output parameter for number of threads.
  * @return True if arguments are valid, false otherwise.
  */
 bool parse_predict_args(int argc, char** argv,
                        std::string& data_file,
                        std::string& model_file,
                        std::string& output_file,
                        int& n_threads) {
     // Initialize with default values
     data_file = "";
     model_file = "";
     output_file = "";
     n_threads = -1;

     // Parse arguments
     for (int i = 1; i < argc; i++) {
         std::string arg = argv[i];

         if (arg == "--data" && i + 1 < argc) {
             data_file = argv[++i];
         } else if (arg == "--model" && i + 1 < argc) {
             model_file = argv[++i];
         } else if (arg == "--out" && i + 1 < argc) {
             output_file = argv[++i];
         } else if (arg == "--nthreads" && i + 1 < argc) {
             n_threads = std::stoi(argv[++i]);
         } else {
             std::cerr << "Unknown option: " << arg << std::endl;
             return false;
         }
     }

     // Validate required arguments
     if (data_file.empty()) {
         std::cerr << "Error: --data is required" << std::endl;
         return false;
     }

     if (model_file.empty()) {
         std::cerr << "Error: --model is required" << std::endl;
         return false;
     }

     if (output_file.empty()) {
         std::cerr << "Error: --out is required" << std::endl;
         return false;
     }

     return true;
 }

 /**
  * @brief Save predictions to a file.
  *
  * @param predictions Prediction values.
  * @param output_file Output file path.
  */
 void save_predictions(const std::vector<float>& predictions, const std::string& output_file) {
     std::ofstream file(output_file);
     if (!file.is_open()) {
         throw std::runtime_error("Unable to open output file: " + output_file);
     }

     for (float pred : predictions) {
         file << pred << std::endl;
     }

     file.close();
 }

 /**
  * @brief Entry point for the predict subcommand.
  *
  * @param argc Number of command-line arguments.
  * @param argv Command-line arguments.
  * @return Exit code.
  */
 int predict_main(int argc, char** argv) {
     if (argc < 2) {
         print_predict_usage();
         return 1;
     }

     // Parse arguments
     std::string data_file;
     std::string model_file;
     std::string output_file;
     int n_threads;

     if (!parse_predict_args(argc, argv, data_file, model_file, output_file, n_threads)) {
         print_predict_usage();
         return 1;
     }

     try {
         // Load model
         std::cout << "Loading model from " << model_file << std::endl;
         boostedpp::GBDT model = boostedpp::load_model_from_json(model_file);

         // Set number of threads if specified
         if (n_threads > 0) {
             model.config().n_threads = n_threads;
         }

         // Load data
         std::cout << "Loading data from " << data_file << std::endl;
         boostedpp::DataMatrix data(data_file, -1); // No label column

         // Make predictions
         std::cout << "Making predictions" << std::endl;
         std::vector<float> predictions = model.predict(data);

         // Save predictions
         std::cout << "Saving predictions to " << output_file << std::endl;
         save_predictions(predictions, output_file);

         std::cout << "Prediction completed successfully" << std::endl;
         return 0;
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 }
