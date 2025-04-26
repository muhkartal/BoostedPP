/**
 * @file tree.hpp
 * @brief Decision tree structures and algorithms for BoostedPP.
 *
 * This file contains the decision tree structures and algorithms used in BoostedPP.
 * It defines the TreeNode and Tree classes that form the building blocks of the GBDT algorithm.
 */

 #ifndef BOOSTEDPP_TREE_HPP
 #define BOOSTEDPP_TREE_HPP

 #include <vector>
 #include <memory>
 #include <limits>
 #include <cstdint>
 #include <string>
 #include <nlohmann/json.hpp>

 #include "data.hpp"
 #include "config.hpp"

 namespace boostedpp {

 /**
  * @brief Structure for split information.
  */
 struct SplitInfo {
     uint32_t feature_id = 0;           ///< Feature ID for the split
     uint32_t bin_id = 0;               ///< Bin ID for the split
     float threshold = 0.0f;            ///< Threshold value for the split
     float gain = -std::numeric_limits<float>::infinity(); ///< Gain for the split
     float left_sum_gradients = 0.0f;   ///< Sum of gradients in the left child
     float left_sum_hessians = 0.0f;    ///< Sum of hessians in the left child
     float right_sum_gradients = 0.0f;  ///< Sum of gradients in the right child
     float right_sum_hessians = 0.0f;   ///< Sum of hessians in the right child

     /**
      * @brief Check if the split is valid.
      * @return True if the split is valid, false otherwise.
      */
     [[nodiscard]] bool is_valid() const noexcept {
         return gain > -std::numeric_limits<float>::infinity();
     }
 };

 /**
  * @brief Structure for a decision tree node.
  */
 struct TreeNode {
     bool is_leaf = true;              ///< Whether the node is a leaf
     uint32_t depth = 0;               ///< Depth of the node
     uint32_t feature_id = 0;          ///< Feature ID for the split
     float threshold = 0.0f;           ///< Threshold value for the split
     float weight = 0.0f;              ///< Weight of the leaf (prediction value)
     uint32_t left_child = 0;          ///< Index of the left child
     uint32_t right_child = 0;         ///< Index of the right child
     float gain = 0.0f;                ///< Gain for the split

     /**
      * @brief Default constructor.
      */
     TreeNode() = default;

     /**
      * @brief Constructor for leaf node.
      * @param depth Node depth.
      * @param weight Leaf weight.
      */
     explicit TreeNode(uint32_t depth, float weight)
         : is_leaf(true), depth(depth), weight(weight) {}

     /**
      * @brief Constructor for internal node.
      * @param depth Node depth.
      * @param feature_id Feature ID for the split.
      * @param threshold Threshold value for the split.
      * @param left_child Index of the left child.
      * @param right_child Index of the right child.
      * @param gain Gain for the split.
      */
     TreeNode(uint32_t depth, uint32_t feature_id, float threshold,
              uint32_t left_child, uint32_t right_child, float gain)
         : is_leaf(false), depth(depth), feature_id(feature_id), threshold(threshold),
           left_child(left_child), right_child(right_child), gain(gain) {}
 };

 /**
  * @brief Class representing a decision tree.
  *
  * This class implements a decision tree that can be used as a weak learner
  * in the gradient boosting algorithm.
  */
 class Tree {
 public:
     /**
      * @brief Default constructor.
      */
     Tree() = default;

     /**
      * @brief Constructor with configuration.
      * @param config GBDT configuration.
      */
     explicit Tree(const GBDTConfig& config);

     /**
      * @brief Build a tree from the given dataset and gradients.
      * @param data Training data.
      * @param gradients First-order gradients.
      * @param hessians Second-order gradients.
      * @param row_indices Indices of rows to include.
      */
     void build(const DataMatrix& data,
               const std::vector<float>& gradients,
               const std::vector<float>& hessians,
               const std::vector<uint32_t>& row_indices);

     /**
      * @brief Predict the output for a single sample.
      * @param features Feature values.
      * @return Predicted value.
      */
     [[nodiscard]] float predict_one(const std::vector<float>& features) const;

     /**
      * @brief Predict the output for a dataset.
      * @param data Test data.
      * @param out_predictions Output predictions.
      */
     void predict(const DataMatrix& data, std::vector<float>& out_predictions) const;

     /**
      * @brief Get the number of nodes in the tree.
      * @return Number of nodes.
      */
     [[nodiscard]] size_t size() const noexcept { return nodes_.size(); }

     /**
      * @brief Convert the tree to XGBoost JSON format.
      * @return JSON representation of the tree.
      */
     [[nodiscard]] nlohmann::json to_xgboost_json() const;

     /**
      * @brief Load a tree from XGBoost JSON format.
      * @param json JSON representation of the tree.
      */
     void from_xgboost_json(const nlohmann::json& json);

     /**
      * @brief Get the nodes of the tree.
      * @return Vector of tree nodes.
      */
     [[nodiscard]] const std::vector<TreeNode>& nodes() const noexcept { return nodes_; }

 private:
     std::vector<TreeNode> nodes_; ///< Vector of tree nodes
     GBDTConfig config_;           ///< GBDT configuration

     /**
      * @brief Find the best split for a node.
      * @param data Training data.
      * @param gradients First-order gradients.
      * @param hessians Second-order gradients.
      * @param row_indices Indices of rows in the node.
      * @param sum_gradients Sum of gradients in the node.
      * @param sum_hessians Sum of hessians in the node.
      * @param hist_gradients Gradient histograms.
      * @param hist_hessians Hessian histograms.
      * @return Best split information.
      */
     [[nodiscard]] SplitInfo find_best_split(
         const DataMatrix& data,
         const std::vector<float>& gradients,
         const std::vector<float>& hessians,
         const std::vector<uint32_t>& row_indices,
         float sum_gradients,
         float sum_hessians,
         std::vector<float>& hist_gradients,
         std::vector<float>& hist_hessians) const;

     /**
      * @brief Split the rows based on the split information.
      * @param data Training data.
      * @param row_indices Indices of rows in the node.
      * @param split Split information.
      * @param left_indices Output indices for the left child.
      * @param right_indices Output indices for the right child.
      */
     void split_rows(
         const DataMatrix& data,
         const std::vector<uint32_t>& row_indices,
         const SplitInfo& split,
         std::vector<uint32_t>& left_indices,
         std::vector<uint32_t>& right_indices) const;

     /**
      * @brief Calculate the leaf weight for a node.
      * @param sum_gradients Sum of gradients in the node.
      * @param sum_hessians Sum of hessians in the node.
      * @return Leaf weight.
      */
     [[nodiscard]] float calculate_leaf_weight(float sum_gradients, float sum_hessians) const;

     /**
      * @brief Build a tree node recursively.
      * @param data Training data.
      * @param gradients First-order gradients.
      * @param hessians Second-order gradients.
      * @param row_indices Indices of rows in the node.
      * @param depth Current depth.
      * @return Index of the built node.
      */
     uint32_t build_node(
         const DataMatrix& data,
         const std::vector<float>& gradients,
         const std::vector<float>& hessians,
         const std::vector<uint32_t>& row_indices,
         uint32_t depth);
 };

 } // namespace boostedpp

 #endif // BOOSTEDPP_TREE_HPP
