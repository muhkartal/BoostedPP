/**
 * @file cv.cpp
 * @brief Implementation of the cross-validation CLI subcommand.
 */

 #include <iostream>
 #include <string>
 #include <unordered_map>
 #include <cstdlib>
 #include <fstream>

 #include "boostedpp/boostedpp.hpp"

 /**
  * @brief Print usage information for the cv subcommand.
  */
 void print_cv_usage() {
     std::cerr << "Usage: boostedpp cv [options]" << std::endl;
     std::cerr << "Options:" << std::endl;
     std::cerr << "  --data <file>       Input data file (CSV format)" << std::endl;
     std::cerr << "  --label <index>     Column index of the label (0-based)" << std::endl;
     std::cerr << "  --folds <int>       Number of folds for cross-validation" << std::endl;
     std::cerr << "  --metric <string>   Evaluation metric (rmse, mae, logloss, auc)" << std::endl;
     std::cerr << "  --nrounds <int>     Number of boosting rounds (default: 100)" << std::endl;
     std::cerr << "  --lr <float>        Learning rate (default: 0.1)" << std::endl;
     std::cerr << "  --max_depth <int>   Maximum depth of trees (default: 6)" << std::endl;
     std::cerr << "  --min_child_weight <float>  Minimum sum of instance weight in a child (default: 1.0)" << std::endl;
     std::cerr << "  --min_data_in_leaf <int>    Minimum number of instances in a leaf (default: 20)" << std::endl;
     std::cerr << "  --reg_lambda <float>        L2 regularization (default: 1.0)" << std::endl;
     std::cerr << "  --subsample <float> Subsample ratio (default: 1.0)" << std::endl;
     std::cerr << "  --colsample <float> Column sample ratio (default: 1.0)" << std::endl;
     std::cerr << "  --nbins <int>       Number of bins for histogram (default: 256)" << std::endl;
     std::cerr << "  --seed <int>        Random seed (default: 0)" << std::endl;
     std::cerr << "  --nthreads <int>    Number of threads (-1 = all) (default: -1)" << std::endl;
     std::cerr << "  --task <task>       Task type (reg = regression, binary = binary classification) (default: reg)" << std::endl;
 }

 /**
  * @brief Parse command line arguments for the cv subcommand.
  *
  * @param argc Number of command-line arguments.
  * @param argv Command-line arguments.
  * @param data_file Output parameter for data file path.
  * @param label_col Output parameter for label column index.
  * @param n_folds Output parameter for number of folds.
  * @param config Output parameter for GBDT configuration.
  * @return True if arguments are valid, false otherwise.
  */
 bool parse_cv_args(int argc, char** argv,
                   std::string& data_file,
                   int& label_col,
                   uint32_t& n_folds,
                   boostedpp::GBDTConfig& config) {
     // Initialize with default values
     data_file = "";
     label_col = -1;
     n_folds = 5;

     // Parse arguments
     for (int i = 1; i < argc; i++) {
         std::string arg = argv[i];

         if (arg == "--data" && i + 1 < argc) {
             data_file = argv[++i];
         } else if (arg == "--label" && i + 1 < argc) {
             label_col = std::stoi(argv[++i]);
         } else if (arg == "--folds" && i + 1 < argc) {
             n_folds = std::stoi(argv[++i]);
         } else if (arg == "--metric" && i + 1 < argc) {
             config.metric = argv[++i];
         } else if (arg == "--nrounds" && i + 1 < argc) {
             config.n_rounds = std::stoi(argv[++i]);
         } else if (arg == "--lr" && i + 1 < argc) {
             config.learning_rate = std::stof(argv[++i]);
         } else if (arg == "--max_depth" && i + 1 < argc) {
             config.max_depth = std::stoi(argv[++i]);
         } else if (arg == "--min_child_weight" && i + 1 < argc) {
             config.min_child_weight = std::stof(argv[++i]);
         } else if (arg == "--min_data_in_leaf" && i + 1 < argc) {
             config.min_data_in_leaf = std::stoi(argv[++i]);
         } else if (arg == "--reg_lambda" && i + 1 < argc) {
             config.reg_lambda = std::stof(argv[++i]);
         } else if (arg == "--subsample" && i + 1 < argc) {
             config.subsample = std::stof(argv[++i]);
         } else if (arg == "--colsample" && i + 1 < argc) {
             config.colsample = std::stof(argv[++i]);
         } else if (arg == "--nbins" && i + 1 < argc) {
             config.n_bins = std::stoi(argv[++i]);
         } else if (arg == "--seed" && i + 1 < argc) {
             config.seed = std::stoi(argv[++i]);
         } else if (arg == "--nthreads" && i + 1 < argc) {
             config.n_threads = std::stoi(argv[++i]);
         } else if (arg == "--task" && i + 1 < argc) {
             std::string task = argv[++i];
             if (task == "reg") {
                 config.task = boostedpp::Task::Regression;
             } else if (task == "binary") {
                 config.task = boostedpp::Task::Binary;
             } else {
                 std::cerr << "Invalid task: " << task << std::endl;
                 return false;
             }
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

     if (label_col < 0) {
         std::cerr << "Error: --label is required" << std::endl;
         return false;
     }

     if (n_folds < 2) {
         std::cerr << "Error: --folds must be at least 2" << std::endl;
         return false;
     }

     // Set default metric if not specified
     if (config.metric.empty()) {
         if (config.task == boostedpp::Task::Binary) {
             config.metric = "logloss";
         } else {
             config.metric = "rmse";
         }
     }

     return true;
 }

 /**
  * @brief Entry point for the cv subcommand.
  *
  * @param argc Number of command-line arguments.
  * @param argv Command-line arguments.
  * @return Exit code.
  */
 int cv_main(int argc, char** argv) {
     if (argc < 2) {
         print_cv_usage();
         return 1;
     }

     // Parse arguments
     std::string data_file;
     int label_col;
     uint32_t n_folds;
     boostedpp::GBDTConfig config;

     if (!parse_cv_args(argc, argv, data_file, label_col, n_folds, config)) {
         print_cv_usage();
         return 1;
     }

     try {
         // Load data
         std::cout << "Loading data from " << data_file << std::endl;
         boostedpp::DataMatrix data(data_file, label_col);

         // Run cross-validation
         std::cout << "Running " << n_folds << "-fold cross-validation with "
                  << config.n_rounds << " boosting rounds" << std::endl;

         boostedpp::GBDT model(config);
         std::vector<float> cv_results = model.cv(data, n_folds);

         // Print results
         std::cout << "Cross-validation results:" << std::endl;
         std::cout << "Rounds\t" << config.metric << std::endl;

         for (uint32_t i = 0; i < cv_results.size(); i++) {
             std::cout << i + 1 << "\t" << cv_results[i] << std::endl;
         }

         std::cout << "Best round: " << std::distance(cv_results.begin(),
                                                   std::min_element(cv_results.begin(), cv_results.end())) + 1
                  << " with " << config.metric << " = " << *std::min_element(cv_results.begin(), cv_results.end())
                  << std::endl;

         std::cout << "Cross-validation completed successfully" << std::endl;
         return 0;
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 }
