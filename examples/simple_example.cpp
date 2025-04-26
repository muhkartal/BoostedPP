/**
 * @file simple_example.cpp
 * @brief A simple example of using the BoostedPP library.
 */

 #include <iostream>
 #include <vector>
 #include <string>
 #include <random>

 #include "boostedpp/boostedpp.hpp"

 /**
  * @brief Generate a simple dataset for demonstration.
  *
  * @param n_samples Number of samples to generate.
  * @param n_features Number of features.
  * @param train_file Path to the training file.
  * @param test_file Path to the test file.
  */
 void generate_dataset(int n_samples, int n_features,
                       const std::string& train_file,
                       const std::string& test_file) {
     // Random number generator
     std::mt19937 gen(42);
     std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

     // Generate training data
     std::ofstream train_out(train_file);
     train_out << "label";
     for (int i = 0; i < n_features; i++) {
         train_out << ",feature" << i;
     }
     train_out << std::endl;

     for (int i = 0; i < n_samples; i++) {
         // Generate features
         std::vector<float> features(n_features);
         for (int j = 0; j < n_features; j++) {
             features[j] = dist(gen);
         }

         // Simple model: label depends on first two features
         float label = features[0] * features[1] > 0 ? 1.0f : 0.0f;

         // Add some noise
         if (dist(gen) > 0.8f) {
             label = 1.0f - label;
         }

         // Write to file
         train_out << label;
         for (float feature : features) {
             train_out << "," << feature;
         }
         train_out << std::endl;
     }
     train_out.close();

     // Generate test data
     std::ofstream test_out(test_file);
     test_out << "label";
     for (int i = 0; i < n_features; i++) {
         test_out << ",feature" << i;
     }
     test_out << std::endl;

     for (int i = 0; i < n_samples / 5; i++) {
         // Generate features
         std::vector<float> features(n_features);
         for (int j = 0; j < n_features; j++) {
             features[j] = dist(gen);
         }

         // Simple model: label depends on first two features
         float label = features[0] * features[1] > 0 ? 1.0f : 0.0f;

         // Add some noise
         if (dist(gen) > 0.8f) {
             label = 1.0f - label;
         }

         // Write to file
         test_out << label;
         for (float feature : features) {
             test_out << "," << feature;
         }
         test_out << std::endl;
     }
     test_out.close();

     std::cout << "Generated dataset with " << n_samples << " training samples and "
               << n_samples / 5 << " test samples" << std::endl;
 }

 int main() {
     // Generate dataset
     generate_dataset(1000, 10, "train.csv", "test.csv");

     try {
         // Load training data
         std::cout << "Loading training data..." << std::endl;
         boostedpp::DataMatrix train_data("train.csv", 0); // 0 is the label column

         // Configure model
         boostedpp::GBDTConfig config;
         config.task = boostedpp::Task::Binary;
         config.n_rounds = 50;
         config.learning_rate = 0.1f;
         config.max_depth = 4;
         config.metric = "logloss";

         // Train model
         std::cout << "Training model..." << std::endl;
         boostedpp::GBDT model(config);
         model.train(train_data);

         // Save model
         std::cout << "Saving model..." << std::endl;
         model.save_model("model.json");

         // Convert to XGBoost format
         model.save_model_to_xgboost_json("model_xgb.json");

         // Load test data
         std::cout << "Loading test data..." << std::endl;
         boostedpp::DataMatrix test_data("test.csv", 0);

         // Make predictions
         std::cout << "Making predictions..." << std::endl;
         std::vector<float> predictions = model.predict(test_data);

         // Calculate AUC
         float auc = boostedpp::auc(test_data.labels(), predictions);
         std::cout << "Test AUC: " << auc << std::endl;

         std::cout << "Example completed successfully!" << std::endl;
         return 0;
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 }
