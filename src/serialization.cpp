/**
 * @file serialization.cpp
 * @brief Implementation of model serialization utilities.
 */

 #include "boostedpp/serialization.hpp"

 #include <fstream>
 #include <stdexcept>
 #include <iostream>

 namespace boostedpp {

 void save_model_to_json(const GBDT& model, const std::string& filename) {
     nlohmann::json model_json;

     // Save configuration
     const auto& config = model.config();
     nlohmann::json config_json;

     config_json["task"] = (config.task == Task::Binary) ? "binary" : "regression";
     config_json["n_rounds"] = config.n_rounds;
     config_json["learning_rate"] = config.learning_rate;
     config_json["max_depth"] = config.max_depth;
     config_json["min_data_in_leaf"] = config.min_data_in_leaf;
     config_json["min_child_weight"] = config.min_child_weight;
     config_json["reg_lambda"] = config.reg_lambda;
     config_json["n_bins"] = config.n_bins;
     config_json["subsample"] = config.subsample;
     config_json["colsample"] = config.colsample;
     config_json["seed"] = config.seed;
     config_json["metric"] = config.metric;

     model_json["config"] = config_json;

     // Save trees
     nlohmann::json trees_json = nlohmann::json::array();

     for (const auto& tree : model.trees()) {
         trees_json.push_back(tree.to_xgboost_json());
     }

     model_json["trees"] = trees_json;

     // Write to file
     std::ofstream file(filename);
     if (!file.is_open()) {
         throw std::runtime_error("Unable to open file for writing: " + filename);
     }

     file << model_json.dump(2);  // Pretty print with 2-space indentation
     file.close();

     std::cout << "Model saved to " << filename << std::endl;
 }

 GBDT load_model_from_json(const std::string& filename) {
     // Read from file
     std::ifstream file(filename);
     if (!file.is_open()) {
         throw std::runtime_error("Unable to open file for reading: " + filename);
     }

     nlohmann::json model_json;
     file >> model_json;
     file.close();

     // Parse configuration
     GBDTConfig config;
     const auto& config_json = model_json["config"];

     config.task = (config_json["task"] == "binary") ? Task::Binary : Task::Regression;
     config.n_rounds = config_json["n_rounds"];
     config.learning_rate = config_json["learning_rate"];
     config.max_depth = config_json["max_depth"];
     config.min_data_in_leaf = config_json["min_data_in_leaf"];
     config.min_child_weight = config_json["min_child_weight"];
     config.reg_lambda = config_json["reg_lambda"];
     config.n_bins = config_json["n_bins"];
     config.subsample = config_json["subsample"];
     config.colsample = config_json["colsample"];
     config.seed = config_json["seed"];
     config.metric = config_json["metric"];

     // Create model
     GBDT model(config);

     // Load trees
     const auto& trees_json = model_json["trees"];
     for (const auto& tree_json : trees_json) {
         Tree tree(config);
         tree.from_xgboost_json(tree_json);
         model.trees().push_back(tree);
     }

     std::cout << "Model loaded from " << filename << std::endl;
     return model;
 }

 nlohmann::json convert_to_xgboost_json(const GBDT& model) {
     nlohmann::json xgb_json;

     // Add model parameters
     const auto& config = model.config();
     nlohmann::json param_json;

     param_json["objective"] = (config.task == Task::Binary) ? "binary:logistic" : "reg:squarederror";
     param_json["eta"] = config.learning_rate;
     param_json["max_depth"] = config.max_depth;
     param_json["min_child_weight"] = config.min_child_weight;
     param_json["lambda"] = config.reg_lambda;
     param_json["subsample"] = config.subsample;
     param_json["colsample_bytree"] = config.colsample;

     xgb_json["learner"] = {
         {"attributes", {{"best_iteration", std::to_string(config.n_rounds)}}},
         {"gradient_booster", {
             {"model", {{"gbtree_model_param", {{"num_trees", config.n_rounds}}}}},
             {"name", "gbtree"}
         }},
         {"learner_model_param", param_json},
         {"name", "generic"},
         {"version", "1.0.0"}
     };

     // Add trees
     auto& trees_json = xgb_json["learner"]["gradient_booster"]["model"]["trees"];
     trees_json = nlohmann::json::array();

     for (const auto& tree : model.trees()) {
         trees_json.push_back(tree.to_xgboost_json());
     }

     return xgb_json;
 }

 GBDT convert_from_xgboost_json(const nlohmann::json& json) {
     // Parse configuration
     GBDTConfig config;

     const auto& learner = json["learner"];
     const auto& params = learner["learner_model_param"];
     const auto& attributes = learner["attributes"];

     // Determine task
     const auto& objective = params["objective"];
     if (objective == "binary:logistic") {
         config.task = Task::Binary;
     } else {
         config.task = Task::Regression;
     }

     // Set parameters
     config.learning_rate = params["eta"];
     config.max_depth = params["max_depth"];
     config.min_child_weight = params["min_child_weight"];
     config.reg_lambda = params["lambda"];
     config.subsample = params["subsample"];
     config.colsample = params["colsample_bytree"];
     config.n_rounds = std::stoi(attributes["best_iteration"].get<std::string>());

     // Create model
     GBDT model(config);

     // Load trees
     const auto& trees_json = learner["gradient_booster"]["model"]["trees"];
     for (const auto& tree_json : trees_json) {
         Tree tree(config);
         tree.from_xgboost_json(tree_json);
         model.trees().push_back(tree);
     }

     return model;
 }

 void save_model_to_xgboost_json(const GBDT& model, const std::string& filename) {
     nlohmann::json xgb_json = convert_to_xgboost_json(model);

     // Write to file
     std::ofstream file(filename);
     if (!file.is_open()) {
         throw std::runtime_error("Unable to open file for writing: " + filename);
     }

     file << xgb_json.dump(2);  // Pretty print with 2-space indentation
     file.close();

     std::cout << "Model saved in XGBoost format to " << filename << std::endl;
 }

 GBDT load_model_from_xgboost_json(const std::string& filename) {
     // Read from file
     std::ifstream file(filename);
     if (!file.is_open()) {
         throw std::runtime_error("Unable to open file for reading: " + filename);
     }

     nlohmann::json xgb_json;
     file >> xgb_json;
     file.close();

     GBDT model = convert_from_xgboost_json(xgb_json);

     std::cout << "Model loaded from XGBoost format " << filename << std::endl;
     return model;
 }

 } // namespace boostedpp
