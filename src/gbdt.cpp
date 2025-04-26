/**
 * @file gbdt.cpp
 * @brief Implementation of the GBDT class.
 */

 #include "boostedpp/gbdt.hpp"

 #include <algorithm>
 #include <numeric>
 #include <stdexcept>
 #include <fstream>
 #include <iostream>
 #include <random>
 #include <omp.h>
 #include <cmath>

 #include "boostedpp/simd_utils.hpp"
 #include "boostedpp/metrics.hpp"
 #include "boostedpp/serialization.hpp"

 namespace boostedpp {

 GBDT::GBDT(const GBDTConfig& config)
     : config_(config) {

     // Validate configuration
     if (!config_.validate()) {
         throw std::invalid_argument("Invalid GBDT configuration");
     }
 }

 void GBDT::train(const DataMatrix& data) {
     if (data.n_rows() == 0 || data.n_cols() == 0) {
         throw std::invalid_argument("Empty dataset");
     }

     if (data.labels().empty()) {
         throw std::invalid_argument("Dataset has no labels");
     }

     // Set number of threads
     if (config_.n_threads > 0) {
         omp_set_num_threads(config_.n_threads);
     }

     // Create bins for histogram-based training
     DataMatrix binned_data = data;
     binned_data.create_bins(config_.n_bins);

     // Initialize predictions with base score
     base_score_ = calculate_base_score(data);
     std::vector<float> predictions(data.n_rows(), base_score_);

     // Initialize gradients and hessians
     std::vector<float> gradients(data.n_rows());
     std::vector<float> hessians(data.n_rows());
     init_gradients(data, gradients, hessians);

     // Prepare row indices (with possible subsampling)
     std::vector<uint32_t> row_indices(data.n_rows());
     std::iota(row_indices.begin(), row_indices.end(), 0);

     // Random generator for subsampling
     std::mt19937 gen(config_.seed);
     std::uniform_real_distribution<float> dist(0.0f, 1.0f);

     // Main training loop
     trees_.clear();
     trees_.reserve(config_.n_rounds);

     for (uint32_t iter = 0; iter < config_.n_rounds; iter++) {
         // Subsample if needed
         std::vector<uint32_t> sampled_indices;
         if (config_.subsample < 1.0f) {
             sampled_indices.reserve(static_cast<size_t>(data.n_rows() * config_.subsample));
             for (uint32_t i = 0; i < data.n_rows(); i++) {
                 if (dist(gen) < config_.subsample) {
                     sampled_indices.push_back(i);
                 }
             }
         } else {
             sampled_indices = row_indices;
         }

         // Create a new tree
         Tree tree(config_);
         tree.build(binned_data, gradients, hessians, sampled_indices);

         // Update predictions and gradients
         update_gradients(data, predictions, tree, gradients, hessians);

         // Add the tree to the ensemble
         trees_.push_back(tree);

         // Calculate and print evaluation metrics
         float eval = evaluate(data, predictions);
         std::cout << "Iteration " << iter << ": " << config_.metric << " = " << eval << std::endl;
     }

     std::cout << "Training completed with " << trees_.size() << " trees" << std::endl;
 }

 std::vector<float> GBDT::predict(const DataMatrix& data) const {
     if (trees_.empty()) {
         throw std::runtime_error("Model is not trained yet");
     }

     // Initialize predictions with base score
     std::vector<float> predictions(data.n_rows(), base_score_);

     // Add contributions from each tree
     for (const auto& tree : trees_) {
         std::vector<float> tree_preds;
         tree.predict(data, tree_preds);

         // Update predictions with learning rate
         for (size_t i = 0; i < predictions.size(); i++) {
             predictions[i] += config_.learning_rate * tree_preds[i];
         }
     }

     // Transform predictions for binary classification
     if (config_.task == Task::Binary) {
         for (size_t i = 0; i < predictions.size(); i++) {
             predictions[i] = 1.0f / (1.0f + std::exp(-predictions[i]));
         }
     }

     return predictions;
 }

 std::vector<float> GBDT::cv(const DataMatrix& data, uint32_t n_folds) const {
     if (data.n_rows() < n_folds) {
         throw std::invalid_argument("Number of folds cannot be greater than number of samples");
     }

     if (data.labels().empty()) {
         throw std::invalid_argument("Dataset has no labels");
     }

     // Get evaluation metric
     auto metric_fn = get_metric(config_.metric);

     // Prepare folds
     std::vector<std::vector<uint32_t>> fold_indices(n_folds);

     // Shuffle indices
     std::vector<uint32_t> indices(data.n_rows());
     std::iota(indices.begin(), indices.end(), 0);
     std::mt19937 gen(config_.seed);
     std::shuffle(indices.begin(), indices.end(), gen);

     // Assign samples to folds
     for (size_t i = 0; i < indices.size(); i++) {
         fold_indices[i % n_folds].push_back(indices[i]);
     }

     // Create a copy of the configuration for training
     GBDTConfig train_config = config_;

     // Prepare output metrics
     std::vector<float> mean_metrics(config_.n_rounds, 0.0f);

     // Run cross-validation
     for (uint32_t fold = 0; fold < n_folds; fold++) {
         // Create train and test sets
         std::vector<uint32_t> train_indices;
         std::vector<uint32_t>& test_indices = fold_indices[fold];

         for (uint32_t i = 0; i < n_folds; i++) {
             if (i != fold) {
                 train_indices.insert(train_indices.end(),
                                     fold_indices[i].begin(),
                                     fold_indices[i].end());
             }
         }

         // Prepare train data
         std::vector<float> train_features;
         std::vector<float> train_labels;

         for (uint32_t idx : train_indices) {
             for (size_t col = 0; col < data.n_cols(); col++) {
                 train_features.push_back(data.get_feature(idx, col));
             }
             train_labels.push_back(data.get_label(idx));
         }

         DataMatrix train_data(train_features, train_labels,
                              train_indices.size(), data.n_cols());

         // Prepare test data
         std::vector<float> test_features;
         std::vector<float> test_labels;

         for (uint32_t idx : test_indices) {
             for (size_t col = 0; col < data.n_cols(); col++) {
                 test_features.push_back(data.get_feature(idx, col));
             }
             test_labels.push_back(data.get_label(idx));
         }

         DataMatrix test_data(test_features, test_labels,
                             test_indices.size(), data.n_cols());

         // Train a model on this fold
         GBDT fold_model(train_config);
         fold_model.train(train_data);

         // Evaluate on test set after each round
         for (uint32_t round = 0; round < config_.n_rounds; round++) {
             // Create a model with first 'round+1' trees
             GBDT eval_model = fold_model;
             eval_model.trees_.resize(round + 1);

             // Make predictions
             std::vector<float> preds = eval_model.predict(test_data);

             // Calculate metric
             float metric_value = metric_fn(test_data.labels(), preds);
             mean_metrics[round] += metric_value / n_folds;
         }
     }

     // Print cross-validation results
     std::cout << "Cross-validation results:" << std::endl;
     for (uint32_t round = 0; round < config_.n_rounds; round++) {
         std::cout << "Round " << round << ": " << config_.metric << " = "
                  << mean_metrics[round] << std::endl;
     }

     return mean_metrics;
 }

 void GBDT::save_model(const std::string& filename) const {
     save_model_to_json(*this, filename);
 }

 void GBDT::load_model(const std::string& filename) {
     *this = load_model_from_json(filename);
 }

 nlohmann::json GBDT::to_xgboost_json() const {
     return convert_to_xgboost_json(*this);
 }

 void GBDT::from_xgboost_json(const nlohmann::json& json) {
     *this = convert_from_xgboost_json(json);
 }

 void GBDT::init_gradients(
     const DataMatrix& data,
     std::vector<float>& out_gradients,
     std::vector<float>& out_hessians) const {

     const float* labels = data.labels().data();
     std::vector<float> preds(data.n_rows(), base_score_);

     if (config_.task == Task::Binary) {
         simd::compute_binary_gradient_hessian(
             labels, preds.data(), data.n_rows(),
             out_gradients.data(), out_hessians.data()
         );
     } else {
         simd::compute_regression_gradient_hessian(
             labels, preds.data(), data.n_rows(),
             out_gradients.data(), out_hessians.data()
         );
     }
 }

 void GBDT::update_gradients(
     const DataMatrix& data,
     std::vector<float>& predictions,
     const Tree& tree,
     std::vector<float>& out_gradients,
     std::vector<float>& out_hessians) const {

     // Get tree predictions
     std::vector<float> tree_preds;
     tree.predict(data, tree_preds);

     // Update predictions with learning rate
     for (size_t i = 0; i < predictions.size(); i++) {
         predictions[i] += config_.learning_rate * tree_preds[i];
     }

     // Calculate new gradients and hessians
     const float* labels = data.labels().data();

     if (config_.task == Task::Binary) {
         simd::compute_binary_gradient_hessian(
             labels, predictions.data(), data.n_rows(),
             out_gradients.data(), out_hessians.data()
         );
     } else {
         simd::compute_regression_gradient_hessian(
             labels, predictions.data(), data.n_rows(),
             out_gradients.data(), out_hessians.data()
         );
     }
 }

 float GBDT::calculate_base_score(const DataMatrix& data) const {
     const auto& labels = data.labels();

     if (config_.task == Task::Binary) {
         // For binary classification, use log-odds of mean label
         float sum = std::accumulate(labels.begin(), labels.end(), 0.0f);
         float mean = sum / labels.size();

         // Clip to avoid extreme values
         mean = std::max(0.01f, std::min(0.99f, mean));

         // Convert to log-odds
         return std::log(mean / (1.0f - mean));
     } else {
         // For regression, use mean label
         float sum = std::accumulate(labels.begin(), labels.end(), 0.0f);
         return sum / labels.size();
     }
 }

 float GBDT::evaluate(
     const DataMatrix& data,
     const std::vector<float>& predictions) const {

     auto metric_fn = get_metric(config_.metric);
     return metric_fn(data.labels(), predictions);
 }

 } // namespace boostedpp
