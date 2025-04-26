/**
 * @file config.hpp
 * @brief Configuration parameters for the BoostedPP GBDT algorithm.
 *
 * This file contains the configuration parameters for the BoostedPP GBDT algorithm.
 * It defines the GBDTConfig struct that holds all parameters used during training.
 */

 #ifndef BOOSTEDPP_CONFIG_HPP
 #define BOOSTEDPP_CONFIG_HPP

 #include <cstdint>
 #include <string>

 namespace boostedpp {

 /**
  * @brief Enum representing the task type (regression or classification).
  */
 enum class Task {
     Regression,   ///< Regression task
     Binary        ///< Binary classification task
 };

 /**
  * @brief Configuration parameters for the GBDT algorithm.
  */
 struct GBDTConfig {
     // Basic parameters
     Task task = Task::Regression;   ///< Task type (regression or binary classification)
     uint32_t n_rounds = 100;        ///< Number of boosting rounds
     float learning_rate = 0.1f;     ///< Learning rate

     // Tree parameters
     uint32_t max_depth = 6;         ///< Maximum depth of trees
     uint32_t min_data_in_leaf = 20; ///< Minimum number of instances in a leaf
     float min_child_weight = 1.0f;  ///< Minimum sum of instance weight in a child
     float reg_lambda = 1.0f;        ///< L2 regularization

     // Histogram parameters
     uint32_t n_bins = 256;          ///< Number of bins for histogram

     // Sampling parameters
     float subsample = 1.0f;         ///< Subsample ratio
     float colsample = 1.0f;         ///< Column sample ratio
     uint32_t seed = 0;              ///< Random seed

     // Parallelization
     int n_threads = -1;             ///< Number of threads (-1 means using all available threads)

     // Metrics
     std::string metric = "rmse";    ///< Evaluation metric

     /**
      * @brief Validate the configuration parameters.
      * @return True if all parameters are valid, false otherwise.
      */
     [[nodiscard]] bool validate() const noexcept;
 };

 } // namespace boostedpp

 #endif // BOOSTEDPP_CONFIG_HPP
