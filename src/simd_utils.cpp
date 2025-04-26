/**
 * @file simd_utils.cpp
 * @brief Implementation of SIMD utilities for optimized operations.
 */

 #include "boostedpp/simd_utils.hpp"

 #include <cmath>
 #include <algorithm>
 #include <stdexcept>
 #include <numeric>

 namespace boostedpp {
 namespace simd {

 #ifdef BOOSTEDPP_USE_AVX2
 // AVX2 implementation

 void compute_histogram(
     const std::vector<uint8_t>& data,
     const std::vector<uint32_t>& row_indices,
     size_t n_rows,
     size_t n_cols,
     uint32_t n_bins,
     std::vector<uint32_t>& out_hist) {

     // Initialize histogram
     out_hist.assign(n_cols * n_bins, 0);

     // Process rows in 8-element chunks where possible
     size_t row_idx = 0;
     const size_t n_rows_aligned = (row_indices.size() / 8) * 8;

     // Process 8 rows at a time
     for (; row_idx < n_rows_aligned; row_idx += 8) {
         // Load 8 row indices
         __m256i row_indices_vec = _mm256_set_epi32(
             row_indices[row_idx + 7],
             row_indices[row_idx + 6],
             row_indices[row_idx + 5],
             row_indices[row_idx + 4],
             row_indices[row_idx + 3],
             row_indices[row_idx + 2],
             row_indices[row_idx + 1],
             row_indices[row_idx]
         );

         // Process each column
         for (size_t col = 0; col < n_cols; col++) {
             // Load bin indices for 8 rows
             uint8_t bin_indices[8];
             for (size_t i = 0; i < 8; i++) {
                 bin_indices[i] = data[row_indices[row_idx + i] * n_cols + col];
             }

             // Update histogram
             for (size_t i = 0; i < 8; i++) {
                 out_hist[col * n_bins + bin_indices[i]]++;
             }
         }
     }

     // Process remaining rows
     for (; row_idx < row_indices.size(); row_idx++) {
         uint32_t row = row_indices[row_idx];
         for (size_t col = 0; col < n_cols; col++) {
             uint8_t bin = data[row * n_cols + col];
             out_hist[col * n_bins + bin]++;
         }
     }
 }

 void compute_gradient_histogram(
     const std::vector<uint8_t>& data,
     const std::vector<uint32_t>& row_indices,
     const std::vector<float>& gradients,
     const std::vector<float>& hessians,
     size_t n_rows,
     size_t n_cols,
     uint32_t n_bins,
     std::vector<float>& grad_hist,
     std::vector<float>& hess_hist) {

     // Initialize histograms
     grad_hist.assign(n_cols * n_bins, 0.0f);
     hess_hist.assign(n_cols * n_bins, 0.0f);

     // Process rows in 8-element chunks where possible
     size_t row_idx = 0;
     const size_t n_rows_aligned = (row_indices.size() / 8) * 8;

     // Process 8 rows at a time
     for (; row_idx < n_rows_aligned; row_idx += 8) {
         // Load 8 gradient values
         __m256 grad_vec = _mm256_set_ps(
             gradients[row_indices[row_idx + 7]],
             gradients[row_indices[row_idx + 6]],
             gradients[row_indices[row_idx + 5]],
             gradients[row_indices[row_idx + 4]],
             gradients[row_indices[row_idx + 3]],
             gradients[row_indices[row_idx + 2]],
             gradients[row_indices[row_idx + 1]],
             gradients[row_indices[row_idx]]
         );

         // Load 8 hessian values
         __m256 hess_vec = _mm256_set_ps(
             hessians[row_indices[row_idx + 7]],
             hessians[row_indices[row_idx + 6]],
             hessians[row_indices[row_idx + 5]],
             hessians[row_indices[row_idx + 4]],
             hessians[row_indices[row_idx + 3]],
             hessians[row_indices[row_idx + 2]],
             hessians[row_indices[row_idx + 1]],
             hessians[row_indices[row_idx]]
         );

         // Process each column
         for (size_t col = 0; col < n_cols; col++) {
             // Load bin indices for 8 rows
             uint8_t bin_indices[8];
             for (size_t i = 0; i < 8; i++) {
                 bin_indices[i] = data[row_indices[row_idx + i] * n_cols + col];
             }

             // Update histograms
             float grad_values[8];
             float hess_values[8];
             _mm256_storeu_ps(grad_values, grad_vec);
             _mm256_storeu_ps(hess_values, hess_vec);

             for (size_t i = 0; i < 8; i++) {
                 uint32_t bin = bin_indices[i];
                 grad_hist[col * n_bins + bin] += grad_values[i];
                 hess_hist[col * n_bins + bin] += hess_values[i];
             }
         }
     }

     // Process remaining rows
     for (; row_idx < row_indices.size(); row_idx++) {
         uint32_t row = row_indices[row_idx];
         float grad = gradients[row];
         float hess = hessians[row];

         for (size_t col = 0; col < n_cols; col++) {
             uint8_t bin = data[row * n_cols + col];
             grad_hist[col * n_bins + bin] += grad;
             hess_hist[col * n_bins + bin] += hess;
         }
     }
 }

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
     float& out_left_sum_h) {

     // Initialize output variables
     out_split_gain = -std::numeric_limits<float>::infinity();
     out_split_bin = 0;
     out_left_sum_g = 0.0f;
     out_left_sum_h = 0.0f;

     // Constants for gain calculation
     const float reg_factor = reg_lambda;
     __m256 reg_vec = _mm256_set1_ps(reg_factor);
     __m256 sum_g_vec = _mm256_set1_ps(sum_gradients);
     __m256 sum_h_vec = _mm256_set1_ps(sum_hessians);
     __m256 min_child_weight_vec = _mm256_set1_ps(min_child_weight);

     // Temporary variables for left sum
     float left_sum_g = 0.0f;
     float left_sum_h = 0.0f;
     __m256 left_sum_g_vec = _mm256_setzero_ps();
     __m256 left_sum_h_vec = _mm256_setzero_ps();

     // Process bins in 8-element chunks
     uint32_t bin_idx = 0;
     const uint32_t n_bins_aligned = (n_bins / 8) * 8;

     for (; bin_idx < n_bins_aligned; bin_idx += 8) {
         // Load gradient and hessian histograms
         __m256 grad_vec = _mm256_loadu_ps(&grad_hist[bin_idx]);
         __m256 hess_vec = _mm256_loadu_ps(&hess_hist[bin_idx]);

         // Update left sums
         left_sum_g_vec = _mm256_add_ps(left_sum_g_vec, grad_vec);
         left_sum_h_vec = _mm256_add_ps(left_sum_h_vec, hess_vec);

         // Calculate right sums
         __m256 right_sum_g_vec = _mm256_sub_ps(sum_g_vec, left_sum_g_vec);
         __m256 right_sum_h_vec = _mm256_sub_ps(sum_h_vec, left_sum_h_vec);

         // Check if both children satisfy the min_child_weight constraint
         __m256 left_weight_check = _mm256_cmp_ps(left_sum_h_vec, min_child_weight_vec, _CMP_GE_OQ);
         __m256 right_weight_check = _mm256_cmp_ps(right_sum_h_vec, min_child_weight_vec, _CMP_GE_OQ);
         __m256 weight_mask = _mm256_and_ps(left_weight_check, right_weight_check);

         // Calculate gains for valid splits
         __m256 left_gain_vec = _mm256_div_ps(
             _mm256_mul_ps(left_sum_g_vec, left_sum_g_vec),
             _mm256_add_ps(left_sum_h_vec, reg_vec)
         );

         __m256 right_gain_vec = _mm256_div_ps(
             _mm256_mul_ps(right_sum_g_vec, right_sum_g_vec),
             _mm256_add_ps(right_sum_h_vec, reg_vec)
         );

         // Compute gain improvement
         __m256 gain_vec = _mm256_add_ps(left_gain_vec, right_gain_vec);
         gain_vec = _mm256_and_ps(gain_vec, weight_mask);

         // Find the best gain in this chunk
         float gains[8];
         _mm256_storeu_ps(gains, gain_vec);

         for (size_t i = 0; i < 8; i++) {
             if (gains[i] > out_split_gain) {
                 out_split_gain = gains[i];
                 out_split_bin = bin_idx + i;

                 // Compute left sums for the best split
                 out_left_sum_g = 0.0f;
                 out_left_sum_h = 0.0f;
                 for (uint32_t j = 0; j <= bin_idx + i; j++) {
                     out_left_sum_g += grad_hist[j];
                     out_left_sum_h += hess_hist[j];
                 }
             }
         }
     }

     // Process remaining bins
     left_sum_g = 0.0f;
     left_sum_h = 0.0f;
     for (uint32_t bin = 0; bin < n_bins; bin++) {
         left_sum_g += grad_hist[bin];
         left_sum_h += hess_hist[bin];

         float right_sum_g = sum_gradients - left_sum_g;
         float right_sum_h = sum_hessians - left_sum_h;

         if (left_sum_h >= min_child_weight && right_sum_h >= min_child_weight) {
             // Calculate gain
             float gain = (left_sum_g * left_sum_g) / (left_sum_h + reg_lambda) +
                          (right_sum_g * right_sum_g) / (right_sum_h + reg_lambda);

             if (gain > out_split_gain) {
                 out_split_gain = gain;
                 out_split_bin = bin;
                 out_left_sum_g = left_sum_g;
                 out_left_sum_h = left_sum_h;
             }
         }
     }
 }

 #elif defined(BOOSTEDPP_USE_SSE42)
 // SSE4.2 implementation

 void compute_histogram(
     const std::vector<uint8_t>& data,
     const std::vector<uint32_t>& row_indices,
     size_t n_rows,
     size_t n_cols,
     uint32_t n_bins,
     std::vector<uint32_t>& out_hist) {

     // Initialize histogram
     out_hist.assign(n_cols * n_bins, 0);

     // Process rows in 4-element chunks where possible
     size_t row_idx = 0;
     const size_t n_rows_aligned = (row_indices.size() / 4) * 4;

     // Process 4 rows at a time
     for (; row_idx < n_rows_aligned; row_idx += 4) {
         // Load 4 row indices
         __m128i row_indices_vec = _mm_set_epi32(
             row_indices[row_idx + 3],
             row_indices[row_idx + 2],
             row_indices[row_idx + 1],
             row_indices[row_idx]
         );

         // Process each column
         for (size_t col = 0; col < n_cols; col++) {
             // Load bin indices for 4 rows
             uint8_t bin_indices[4];
             for (size_t i = 0; i < 4; i++) {
                 bin_indices[i] = data[row_indices[row_idx + i] * n_cols + col];
             }

             // Update histogram
             for (size_t i = 0; i < 4; i++) {
                 out_hist[col * n_bins + bin_indices[i]]++;
             }
         }
     }

     // Process remaining rows
     for (; row_idx < row_indices.size(); row_idx++) {
         uint32_t row = row_indices[row_idx];
         for (size_t col = 0; col < n_cols; col++) {
             uint8_t bin = data[row * n_cols + col];
             out_hist[col * n_bins + bin]++;
         }
     }
 }

 void compute_gradient_histogram(
     const std::vector<uint8_t>& data,
     const std::vector<uint32_t>& row_indices,
     const std::vector<float>& gradients,
     const std::vector<float>& hessians,
     size_t n_rows,
     size_t n_cols,
     uint32_t n_bins,
     std::vector<float>& grad_hist,
     std::vector<float>& hess_hist) {

     // Initialize histograms
     grad_hist.assign(n_cols * n_bins, 0.0f);
     hess_hist.assign(n_cols * n_bins, 0.0f);

     // Process rows in 4-element chunks where possible
     size_t row_idx = 0;
     const size_t n_rows_aligned = (row_indices.size() / 4) * 4;

     // Process 4 rows at a time
     for (; row_idx < n_rows_aligned; row_idx += 4) {
         // Load 4 gradient values
         __m128 grad_vec = _mm_set_ps(
             gradients[row_indices[row_idx + 3]],
             gradients[row_indices[row_idx + 2]],
             gradients[row_indices[row_idx + 1]],
             gradients[row_indices[row_idx]]
         );

         // Load 4 hessian values
         __m128 hess_vec = _mm_set_ps(
             hessians[row_indices[row_idx + 3]],
             hessians[row_indices[row_idx + 2]],
             hessians[row_indices[row_idx + 1]],
             hessians[row_indices[row_idx]]
         );

         // Process each column
         for (size_t col = 0; col < n_cols; col++) {
             // Load bin indices for 4 rows
             uint8_t bin_indices[4];
             for (size_t i = 0; i < 4; i++) {
                 bin_indices[i] = data[row_indices[row_idx + i] * n_cols + col];
             }

             // Update histograms
             float grad_values[4];
             float hess_values[4];
             _mm_storeu_ps(grad_values, grad_vec);
             _mm_storeu_ps(hess_values, hess_vec);

             for (size_t i = 0; i < 4; i++) {
                 uint32_t bin = bin_indices[i];
                 grad_hist[col * n_bins + bin] += grad_values[i];
                 hess_hist[col * n_bins + bin] += hess_values[i];
             }
         }
     }

     // Process remaining rows
     for (; row_idx < row_indices.size(); row_idx++) {
         uint32_t row = row_indices[row_idx];
         float grad = gradients[row];
         float hess = hessians[row];

         for (size_t col = 0; col < n_cols; col++) {
             uint8_t bin = data[row * n_cols + col];
             grad_hist[col * n_bins + bin] += grad;
             hess_hist[col * n_bins + bin] += hess;
         }
     }
 }

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
     float& out_left_sum_h) {

     // Initialize output variables
     out_split_gain = -std::numeric_limits<float>::infinity();
     out_split_bin = 0;
     out_left_sum_g = 0.0f;
     out_left_sum_h = 0.0f;

     // Constants for gain calculation
     const float reg_factor = reg_lambda;
     __m128 reg_vec = _mm_set1_ps(reg_factor);
     __m128 sum_g_vec = _mm_set1_ps(sum_gradients);
     __m128 sum_h_vec = _mm_set1_ps(sum_hessians);
     __m128 min_child_weight_vec = _mm_set1_ps(min_child_weight);

     // Temporary variables for left sum
     float left_sum_g = 0.0f;
     float left_sum_h = 0.0f;
     __m128 left_sum_g_vec = _mm_setzero_ps();
     __m128 left_sum_h_vec = _mm_setzero_ps();

     // Process bins in 4-element chunks
     uint32_t bin_idx = 0;
     const uint32_t n_bins_aligned = (n_bins / 4) * 4;

     for (; bin_idx < n_bins_aligned; bin_idx += 4) {
         // Load gradient and hessian histograms
         __m128 grad_vec = _mm_loadu_ps(&grad_hist[bin_idx]);
         __m128 hess_vec = _mm_loadu_ps(&hess_hist[bin_idx]);

         // Update left sums
         left_sum_g_vec = _mm_add_ps(left_sum_g_vec, grad_vec);
         left_sum_h_vec = _mm_add_ps(left_sum_h_vec, hess_vec);

         // Calculate right sums
         __m128 right_sum_g_vec = _mm_sub_ps(sum_g_vec, left_sum_g_vec);
         __m128 right_sum_h_vec = _mm_sub_ps(sum_h_vec, left_sum_h_vec);

         // Check if both children satisfy the min_child_weight constraint
         __m128 left_weight_check = _mm_cmpge_ps(left_sum_h_vec, min_child_weight_vec);
         __m128 right_weight_check = _mm_cmpge_ps(right_sum_h_vec, min_child_weight_vec);
         __m128 weight_mask = _mm_and_ps(left_weight_check, right_weight_check);

         // Calculate gains for valid splits
         __m128 left_gain_vec = _mm_div_ps(
             _mm_mul_ps(left_sum_g_vec, left_sum_g_vec),
             _mm_add_ps(left_sum_h_vec, reg_vec)
         );

         __m128 right_gain_vec = _mm_div_ps(
             _mm_mul_ps(right_sum_g_vec, right_sum_g_vec),
             _mm_add_ps(right_sum_h_vec, reg_vec)
         );

         // Compute gain improvement
         __m128 gain_vec = _mm_add_ps(left_gain_vec, right_gain_vec);
         gain_vec = _mm_and_ps(gain_vec, weight_mask);

         // Find the best gain in this chunk
         float gains[4];
         _mm_storeu_ps(gains, gain_vec);

         for (size_t i = 0; i < 4; i++) {
             if (gains[i] > out_split_gain) {
                 out_split_gain = gains[i];
                 out_split_bin = bin_idx + i;

                 // Compute left sums for the best split
                 out_left_sum_g = 0.0f;
                 out_left_sum_h = 0.0f;
                 for (uint32_t j = 0; j <= bin_idx + i; j++) {
                     out_left_sum_g += grad_hist[j];
                     out_left_sum_h += hess_hist[j];
                 }
             }
         }
     }

     // Process remaining bins
     left_sum_g = 0.0f;
     left_sum_h = 0.0f;
     for (uint32_t bin = 0; bin < n_bins; bin++) {
         left_sum_g += grad_hist[bin];
         left_sum_h += hess_hist[bin];

         float right_sum_g = sum_gradients - left_sum_g;
         float right_sum_h = sum_hessians - left_sum_h;

         if (left_sum_h >= min_child_weight && right_sum_h >= min_child_weight) {
             // Calculate gain
             float gain = (left_sum_g * left_sum_g) / (left_sum_h + reg_lambda) +
                          (right_sum_g * right_sum_g) / (right_sum_h + reg_lambda);

             if (gain > out_split_gain) {
                 out_split_gain = gain;
                 out_split_bin = bin;
                 out_left_sum_g = left_sum_g;
                 out_left_sum_h = left_sum_h;
             }
         }
     }
 }

 #else
 // Fallback implementation (no SIMD)

 void compute_histogram(
     const std::vector<uint8_t>& data,
     const std::vector<uint32_t>& row_indices,
     size_t n_rows,
     size_t n_cols,
     uint32_t n_bins,
     std::vector<uint32_t>& out_hist) {

     // Initialize histogram
     out_hist.assign(n_cols * n_bins, 0);

     // Process each row
     for (uint32_t row_idx : row_indices) {
         for (size_t col = 0; col < n_cols; col++) {
             uint8_t bin = data[row_idx * n_cols + col];
             out_hist[col * n_bins + bin]++;
         }
     }
 }

 void compute_gradient_histogram(
     const std::vector<uint8_t>& data,
     const std::vector<uint32_t>& row_indices,
     const std::vector<float>& gradients,
     const std::vector<float>& hessians,
     size_t n_rows,
     size_t n_cols,
     uint32_t n_bins,
     std::vector<float>& grad_hist,
     std::vector<float>& hess_hist) {

     // Initialize histograms
     grad_hist.assign(n_cols * n_bins, 0.0f);
     hess_hist.assign(n_cols * n_bins, 0.0f);

     // Process each row
     for (uint32_t row_idx : row_indices) {
         float grad = gradients[row_idx];
         float hess = hessians[row_idx];

         for (size_t col = 0; col < n_cols; col++) {
             uint8_t bin = data[row_idx * n_cols + col];
             grad_hist[col * n_bins + bin] += grad;
             hess_hist[col * n_bins + bin] += hess;
         }
     }
 }

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
     float& out_left_sum_h) {

     // Initialize output variables
     out_split_gain = -std::numeric_limits<float>::infinity();
     out_split_bin = 0;
     out_left_sum_g = 0.0f;
     out_left_sum_h = 0.0f;

     // Process each bin
     float left_sum_g = 0.0f;
     float left_sum_h = 0.0f;

     for (uint32_t bin = 0; bin < n_bins; bin++) {
         left_sum_g += grad_hist[bin];
         left_sum_h += hess_hist[bin];

         float right_sum_g = sum_gradients - left_sum_g;
         float right_sum_h = sum_hessians - left_sum_h;

         if (left_sum_h >= min_child_weight && right_sum_h >= min_child_weight) {
             // Calculate gain
             float gain = (left_sum_g * left_sum_g) / (left_sum_h + reg_lambda) +
                          (right_sum_g * right_sum_g) / (right_sum_h + reg_lambda) -
                          (sum_gradients * sum_gradients) / (sum_hessians + reg_lambda);

             if (gain > out_split_gain) {
                 out_split_gain = gain;
                 out_split_bin = bin;
                 out_left_sum_g = left_sum_g;
                 out_left_sum_h = left_sum_h;
             }
         }
     }
 }

 #endif

 // Common implementations (same for all SIMD variants)

 void compute_binary_gradient_hessian(
     const float* labels,
     const float* preds,
     size_t n_rows,
     float* out_gradients,
     float* out_hessians) {

     #pragma omp parallel for
     for (size_t i = 0; i < n_rows; i++) {
         float p = 1.0f / (1.0f + std::exp(-preds[i]));
         out_gradients[i] = p - labels[i];
         out_hessians[i] = p * (1.0f - p);
     }
 }

 void compute_regression_gradient_hessian(
     const float* labels,
     const float* preds,
     size_t n_rows,
     float* out_gradients,
     float* out_hessians) {

     #pragma omp parallel for
     for (size_t i = 0; i < n_rows; i++) {
         out_gradients[i] = preds[i] - labels[i];
         out_hessians[i] = 1.0f;
     }
 }

 const char* get_simd_instruction_set() {
 #ifdef BOOSTEDPP_USE_AVX2
     return "AVX2";
 #elif defined(BOOSTEDPP_USE_SSE42)
     return "SSE4.2";
 #else
     return "Scalar (no SIMD)";
 #endif
 }

 } // namespace simd
 } // namespace boostedpp
