/**
 * @file tree.cpp
 * @brief Implementation of the Tree class.
 */

 #include "boostedpp/tree.hpp"

 #include <algorithm>
 #include <numeric>
 #include <stdexcept>
 #include <omp.h>
 #include <iostream>
 #include <queue>

 #include "boostedpp/simd_utils.hpp"

 namespace boostedpp {

 Tree::Tree(const GBDTConfig& config)
     : config_(config) {
 }

 void Tree::build(const DataMatrix& data,
                 const std::vector<float>& gradients,
                 const std::vector<float>& hessians,
                 const std::vector<uint32_t>& row_indices) {

     // Clear any existing nodes
     nodes_.clear();

     // Build the tree recursively starting from the root
     build_node(data, gradients, hessians, row_indices, 0);

     std::cout << "Built tree with " << nodes_.size() << " nodes" << std::endl;
 }

 float Tree::predict_one(const std::vector<float>& features) const {
     if (nodes_.empty()) {
         throw std::runtime_error("Tree is not built yet");
     }

     size_t node_idx = 0;
     while (!nodes_[node_idx].is_leaf) {
         const TreeNode& node = nodes_[node_idx];
         uint32_t feature_id = node.feature_id;
         float threshold = node.threshold;

         if (std::isnan(features[feature_id])) {
             // Handle missing value
             // In this simple implementation, we always go to the right child
             node_idx = node.right_child;
         } else if (features[feature_id] <= threshold) {
             node_idx = node.left_child;
         } else {
             node_idx = node.right_child;
         }
     }

     return nodes_[node_idx].weight;
 }

 void Tree::predict(const DataMatrix& data, std::vector<float>& out_predictions) const {
     if (nodes_.empty()) {
         throw std::runtime_error("Tree is not built yet");
     }

     size_t n_rows = data.n_rows();
     out_predictions.resize(n_rows);

     #pragma omp parallel for schedule(static)
     for (size_t i = 0; i < n_rows; i++) {
         // Extract features for this row
         std::vector<float> features(data.n_cols());
         for (size_t j = 0; j < data.n_cols(); j++) {
             features[j] = data.get_feature(i, j);
         }

         // Predict
         out_predictions[i] = predict_one(features);
     }
 }

 nlohmann::json Tree::to_xgboost_json() const {
     nlohmann::json tree_json;

     // Create a map of node indices to XGBoost node indices
     std::unordered_map<uint32_t, uint32_t> node_map;
     uint32_t next_node_id = 0;

     // Process nodes breadth-first
     std::queue<uint32_t> queue;
     queue.push(0); // Start with root

     while (!queue.empty()) {
         uint32_t node_idx = queue.front();
         queue.pop();

         // Assign XGBoost node ID
         node_map[node_idx] = next_node_id++;

         // Add children to queue
         const TreeNode& node = nodes_[node_idx];
         if (!node.is_leaf) {
             queue.push(node.left_child);
             queue.push(node.right_child);
         }
     }

     // Build node list
     nlohmann::json nodes_json = nlohmann::json::array();

     for (size_t i = 0; i < nodes_.size(); i++) {
         const TreeNode& node = nodes_[i];
         nlohmann::json node_json;

         if (node.is_leaf) {
             node_json["nodeid"] = node_map[i];
             node_json["leaf"] = node.weight;
         } else {
             node_json["nodeid"] = node_map[i];
             node_json["split"] = node.feature_id;
             node_json["split_condition"] = node.threshold;
             node_json["yes"] = node_map[node.left_child];
             node_json["no"] = node_map[node.right_child];
             node_json["missing"] = node_map[node.right_child]; // Same as no
         }

         nodes_json.push_back(node_json);
     }

     tree_json["nodes"] = nodes_json;

     return tree_json;
 }

 void Tree::from_xgboost_json(const nlohmann::json& json) {
     if (!json.contains("nodes")) {
         throw std::runtime_error("Invalid XGBoost tree format: missing 'nodes'");
     }

     const auto& nodes_json = json["nodes"];
     if (!nodes_json.is_array()) {
         throw std::runtime_error("Invalid XGBoost tree format: 'nodes' is not an array");
     }

     // First pass: count nodes and create a mapping from XGBoost node IDs to our node indices
     std::unordered_map<uint32_t, uint32_t> node_map;
     for (const auto& node_json : nodes_json) {
         uint32_t node_id = node_json["nodeid"];
         node_map[node_id] = static_cast<uint32_t>(node_map.size());
     }

     // Resize our nodes vector
     nodes_.resize(node_map.size());

     // Second pass: fill node data
     for (const auto& node_json : nodes_json) {
         uint32_t xgb_node_id = node_json["nodeid"];
         uint32_t our_node_id = node_map[xgb_node_id];

         if (node_json.contains("leaf")) {
             // Leaf node
             nodes_[our_node_id].is_leaf = true;
             nodes_[our_node_id].weight = node_json["leaf"];
         } else {
             // Internal node
             nodes_[our_node_id].is_leaf = false;
             nodes_[our_node_id].feature_id = node_json["split"];
             nodes_[our_node_id].threshold = node_json["split_condition"];
             nodes_[our_node_id].left_child = node_map[static_cast<uint32_t>(node_json["yes"])];
             nodes_[our_node_id].right_child = node_map[static_cast<uint32_t>(node_json["no"])];

             // Calculate node depth
             // In this simple implementation, we leave depth as 0
             // A more complete implementation would calculate the depth in a separate pass
         }
     }
 }

 SplitInfo Tree::find_best_split(
     const DataMatrix& data,
     const std::vector<float>& gradients,
     const std::vector<float>& hessians,
     const std::vector<uint32_t>& row_indices,
     float sum_gradients,
     float sum_hessians,
     std::vector<float>& hist_gradients,
     std::vector<float>& hist_hessians) const {

     SplitInfo best_split;
     best_split.gain = -std::numeric_limits<float>::infinity();

     const size_t n_cols = data.n_cols();
     const uint32_t n_bins = config_.n_bins;

     // Prepare for parallel processing
     std::vector<SplitInfo> feature_best_splits(n_cols);

     // Process each feature in parallel
     #pragma omp parallel for schedule(dynamic, 1)
     for (size_t feature_id = 0; feature_id < n_cols; feature_id++) {
         // Extract histograms for this feature
         std::vector<float> feature_grad_hist(n_bins, 0.0f);
         std::vector<float> feature_hess_hist(n_bins, 0.0f);

         // Compute gradient histograms for this feature
         for (uint32_t row_idx : row_indices) {
             uint8_t bin = data.get_binned_feature(row_idx, feature_id);
             feature_grad_hist[bin] += gradients[row_idx];
             feature_hess_hist[bin] += hessians[row_idx];
         }

         // Find best split for this feature
         float split_gain;
         uint32_t split_bin;
         float left_sum_g, left_sum_h;

         simd::find_best_split(
             feature_grad_hist, feature_hess_hist, n_bins,
             sum_gradients, sum_hessians, config_.min_child_weight, config_.reg_lambda,
             split_gain, split_bin, left_sum_g, left_sum_h
         );

         // Store the best split for this feature
         feature_best_splits[feature_id].feature_id = static_cast<uint32_t>(feature_id);
         feature_best_splits[feature_id].bin_id = split_bin;
         feature_best_splits[feature_id].gain = split_gain;
         feature_best_splits[feature_id].left_sum_gradients = left_sum_g;
         feature_best_splits[feature_id].left_sum_hessians = left_sum_h;
         feature_best_splits[feature_id].right_sum_gradients = sum_gradients - left_sum_g;
         feature_best_splits[feature_id].right_sum_hessians = sum_hessians - left_sum_h;

         // Convert bin ID to threshold
         if (feature_best_splits[feature_id].gain > -std::numeric_limits<float>::infinity()) {
             const BinInfo& bin_info = data.bin_info()[feature_id];
             if (split_bin < bin_info.splits.size()) {
                 feature_best_splits[feature_id].threshold = bin_info.splits[split_bin];
             } else {
                 // This should not happen in normal operation
                 feature_best_splits[feature_id].threshold = std::numeric_limits<float>::quiet_NaN();
             }
         }
     }

     // Find the best split across all features
     for (const auto& split : feature_best_splits) {
         if (split.gain > best_split.gain) {
             best_split = split;
         }
     }

     return best_split;
 }

 void Tree::split_rows(
     const DataMatrix& data,
     const std::vector<uint32_t>& row_indices,
     const SplitInfo& split,
     std::vector<uint32_t>& left_indices,
     std::vector<uint32_t>& right_indices) const {

     left_indices.clear();
     right_indices.clear();
     left_indices.reserve(row_indices.size());
     right_indices.reserve(row_indices.size());

     for (uint32_t row_idx : row_indices) {
         float value = data.get_feature(row_idx, split.feature_id);

         if (std::isnan(value)) {
             // Handle missing value (go to right child)
             right_indices.push_back(row_idx);
         } else if (value <= split.threshold) {
             left_indices.push_back(row_idx);
         } else {
             right_indices.push_back(row_idx);
         }
     }
 }

 float Tree::calculate_leaf_weight(float sum_gradients, float sum_hessians) const {
     return -sum_gradients / (sum_hessians + config_.reg_lambda);
 }

 uint32_t Tree::build_node(
     const DataMatrix& data,
     const std::vector<float>& gradients,
     const std::vector<float>& hessians,
     const std::vector<uint32_t>& row_indices,
     uint32_t depth) {

     // Calculate sum of gradients and hessians
     float sum_gradients = 0.0f;
     float sum_hessians = 0.0f;

     for (uint32_t row_idx : row_indices) {
         sum_gradients += gradients[row_idx];
         sum_hessians += hessians[row_idx];
     }

     // Create a leaf node if we've reached maximum depth or minimum child weight
     if (depth >= config_.max_depth || sum_hessians < config_.min_child_weight || row_indices.size() <= config_.min_data_in_leaf) {
         // Create a leaf node
         float weight = calculate_leaf_weight(sum_gradients, sum_hessians);
         nodes_.emplace_back(depth, weight);
         return static_cast<uint32_t>(nodes_.size() - 1);
     }

     // Find the best split
     std::vector<float> hist_gradients;
     std::vector<float> hist_hessians;

     SplitInfo best_split = find_best_split(
         data, gradients, hessians, row_indices,
         sum_gradients, sum_hessians, hist_gradients, hist_hessians
     );

     // Create a leaf node if no valid split found
     if (!best_split.is_valid()) {
         float weight = calculate_leaf_weight(sum_gradients, sum_hessians);
         nodes_.emplace_back(depth, weight);
         return static_cast<uint32_t>(nodes_.size() - 1);
     }

     // Split the data
     std::vector<uint32_t> left_indices, right_indices;
     split_rows(data, row_indices, best_split, left_indices, right_indices);

     // Check if either child is empty
     if (left_indices.empty() || right_indices.empty()) {
         float weight = calculate_leaf_weight(sum_gradients, sum_hessians);
         nodes_.emplace_back(depth, weight);
         return static_cast<uint32_t>(nodes_.size() - 1);
     }

     // Create an internal node
     uint32_t node_idx = static_cast<uint32_t>(nodes_.size());
     nodes_.emplace_back(); // Placeholder

     // Build children
     uint32_t left_idx = build_node(data, gradients, hessians, left_indices, depth + 1);
     uint32_t right_idx = build_node(data, gradients, hessians, right_indices, depth + 1);

     // Update the internal node
     nodes_[node_idx] = TreeNode(depth, best_split.feature_id, best_split.threshold,
                                left_idx, right_idx, best_split.gain);

     return node_idx;
 }

 } // namespace boostedpp
