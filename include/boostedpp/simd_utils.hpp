/**
 * @file simd_utils.hpp
 * @brief SIMD utilities for optimized operations in BoostedPP.
 *
 * This file contains SIMD utilities for vectorized gain calculations
 * and other performance-critical operations in BoostedPP.
 */

 #ifndef BOOSTEDPP_SIMD_UTILS_HPP
 #define BOOSTEDPP_SIMD_UTILS_HPP

 #include <cstdint>
 #include <vector>
 #include <array>

 // Check for available SIMD instruction sets
 #if defined(__AVX2__)
     #include <immintrin.h>
     #define BOOSTEDPP_USE_AVX2
 #elif defined(__SSE4_2__)
     #include <smmintrin.h>
     #define BOOSTEDPP_USE_SSE42
 #endif

 namespace boostedpp {
 namespace simd {

 /**
  * @brief Compute histogram for a subset of data using SIMD instructions.
  *
  * @param data Binned data (row-major)
  * @param row_indices Indices of rows to include
  * @param n_rows Number of rows
  * @param n_cols Number of columns
  * @param n_bins Number of bins per feature
  * @param out_hist Output histogram array (size = n_cols * n_bins)
  */
 void compute_histogram(
     const std::vector<uint8_t>& data,
     const std::vector<uint32_t>& row_indices,
     size_t n_rows,
     size_t n_cols,
     uint32_t n_bins,
     std::vector<uint32_t>& out_hist);

 /**
  * @brief Compute gradient histograms for a subset of data using SIMD instructions.
  *
  * @param data Binned data (row-major)
  * @param row_indices Indices of rows to include
  * @param gradients First-order gradients
  * @param hessians Second-order gradients
  * @param n_rows Number of rows
  * @param n_cols Number of columns
  * @param n_bins Number of bins per feature
  * @param grad_hist Output gradient histogram array (size = n_cols * n_bins)
  * @param hess_hist Output hessian histogram array (size = n_cols * n_bins)
  */
 void compute_gradient_histogram(
     const std::vector<uint8_t>& data,
     const std::vector<uint32_t>& row_indices,
     const std::vector<float>& gradients,
     const std::vector<float>& hessians,
     size_t n_rows,
     size_t n_cols,
     uint32_t n_bins,
     std::vector<float>& grad_hist,
     std::vector<float>& hess_hist);

 /**
  * @brief Compute the best split for a feature using SIMD instructions.
  *
  * @param grad_hist Gradient histogram for the feature
  * @param hess_hist Hessian histogram for the feature
  * @param n_bins Number of bins
  * @param sum_gradients Sum of gradients in the node
  * @param sum_hessians Sum of hessians in the node
  * @param min_child_weight Minimum sum of hessian in a child
  * @param reg_lambda L2 regularization
  * @param out_split_gain Best split gain
  * @param out_split_bin Best split bin
  * @param out_left_sum_g Sum of gradients in the left child
  * @param out_left_sum_h Sum of hessians in the left child
  */
 void find_best_split(
     const std::vector<float>& grad_hist,
     const std::vector<float>& hess_hist,
     uint32_t n_bins,
     float sum_gradients,
     float sum_hessians,
     float min_child_weight,
     float reg_lambda,
     float& out_split_gain,
     uint32_t& out_split_bin,
     float& out_left_sum_g,
     float& out_left_sum_h);

 /**
  * @brief Compute gradient and hessian for binary classification.
  *
  * @param labels True labels
  * @param preds Predicted values
  * @param n_rows Number of rows
  * @param out_gradients Output gradients
  * @param out_hessians Output hessians
  */
 void compute_binary_gradient_hessian(
     const float* labels,
     const float* preds,
     size_t n_rows,
     float* out_gradients,
     float* out_hessians);

 /**
  * @brief Compute gradient and hessian for regression.
  *
  * @param labels True labels
  * @param preds Predicted values
  * @param n_rows Number of rows
  * @param out_gradients Output gradients
  * @param out_hessians Output hessians
  */
 void compute_regression_gradient_hessian(
     const float* labels,
     const float* preds,
     size_t n_rows,
     float* out_gradients,
     float* out_hessians);

 /**
  * @brief Get the SIMD instruction set being used.
  *
  * @return String representation of the SIMD instruction set.
  */
 const char* get_simd_instruction_set();

 } // namespace simd
 } // namespace boostedpp

 #endif // BOOSTEDPP_SIMD_UTILS_HPP
