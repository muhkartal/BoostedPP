/**
 * @file data.cpp
 * @brief Implementation of the DataMatrix class.
 */

 #include "boostedpp/data.hpp"

 #include <algorithm>
 #include <fstream>
 #include <sstream>
 #include <stdexcept>
 #include <limits>
 #include <cmath>
 #include <iostream>
 #include <numeric>

 namespace boostedpp {

 uint32_t BinInfo::get_bin(float value) const noexcept {
     // Handle missing values
     if (std::isnan(value)) {
         return static_cast<uint32_t>(splits.size()); // Last bin is reserved for missing values
     }

     // Find the appropriate bin using binary search
     auto it = std::upper_bound(splits.begin(), splits.end(), value);
     return static_cast<uint32_t>(std::distance(splits.begin(), it));
 }

 DataMatrix::DataMatrix(const std::string& filename, int label_column) {
     load_csv(filename, label_column);
 }

 DataMatrix::DataMatrix(const std::vector<float>& features, const std::vector<float>& labels,
                        size_t n_rows, size_t n_cols)
     : n_rows_(n_rows), n_cols_(n_cols), features_(features), labels_(labels) {

     if (features.size() != n_rows * n_cols) {
         throw std::invalid_argument(
             "Feature vector size does not match n_rows * n_cols");
     }

     if (!labels.empty() && labels.size() != n_rows) {
         throw std::invalid_argument(
             "Label vector size does not match n_rows");
     }
 }

 void DataMatrix::load_csv(const std::string& filename, int label_column) {
     std::ifstream file(filename);
     if (!file.is_open()) {
         throw std::runtime_error("Unable to open file: " + filename);
     }

     // Read the header
     std::string line;
     std::getline(file, line);
     std::stringstream ss(line);
     std::string cell;
     std::vector<std::string> header;

     while (std::getline(ss, cell, ',')) {
         header.push_back(cell);
     }

     // Determine number of columns
     n_cols_ = header.size();
     if (label_column >= 0) {
         n_cols_--; // Exclude label column from feature count
     }

     // Temporary storage for raw data
     std::vector<std::vector<float>> raw_features;
     std::vector<float> raw_labels;

     // Read the data
     while (std::getline(file, line)) {
         std::stringstream ss(line);
         std::vector<float> row;
         int col_idx = 0;

         while (std::getline(ss, cell, ',')) {
             float value;
             try {
                 // Check for missing values
                 if (cell.empty() || cell == "NA" || cell == "N/A" || cell == "?") {
                     value = kMissingValue;
                 } else {
                     value = std::stof(cell);
                 }

                 // Store as label or feature
                 if (col_idx == label_column) {
                     raw_labels.push_back(value);
                 } else {
                     row.push_back(value);
                 }
             } catch (const std::exception& e) {
                 throw std::runtime_error(
                     "Error parsing value at row " + std::to_string(raw_features.size() + 1) +
                     ", col " + std::to_string(col_idx) + ": " + e.what());
             }

             col_idx++;
         }

         if (row.size() != n_cols_) {
             throw std::runtime_error(
                 "Inconsistent number of columns at row " + std::to_string(raw_features.size() + 1));
         }

         raw_features.push_back(row);
     }

     // Set number of rows
     n_rows_ = raw_features.size();

     // Validate label column if specified
     if (label_column >= 0 && raw_labels.size() != n_rows_) {
         throw std::runtime_error("Inconsistent number of labels");
     }

     // Convert to row-major storage
     features_.resize(n_rows_ * n_cols_);
     for (size_t i = 0; i < n_rows_; i++) {
         for (size_t j = 0; j < n_cols_; j++) {
             features_[i * n_cols_ + j] = raw_features[i][j];
         }
     }

     // Store labels
     labels_ = raw_labels;

     std::cout << "Loaded " << n_rows_ << " rows and " << n_cols_ << " columns from " << filename << std::endl;
 }

 void DataMatrix::create_bins(uint32_t n_bins) {
     // Initialize bin information
     bin_info_.resize(n_cols_);

     // Process each feature
     for (size_t col = 0; col < n_cols_; col++) {
         // Extract non-missing values for this feature
         std::vector<float> values;
         values.reserve(n_rows_);

         for (size_t row = 0; row < n_rows_; row++) {
             float val = features_[row * n_cols_ + col];
             if (!std::isnan(val)) {
                 values.push_back(val);
             }
         }

         // Sort values
         std::sort(values.begin(), values.end());

         // Remove duplicates
         auto last = std::unique(values.begin(), values.end());
         values.erase(last, values.end());

         // Create bins
         bin_info_[col].type = BinType::Numerical;

         if (values.size() <= n_bins) {
             // If there are fewer unique values than bins, use all values
             bin_info_[col].splits = values;
         } else {
             // Otherwise, create equally-sized bins
             bin_info_[col].splits.resize(n_bins - 1); // Reserve last bin for missing values

             for (uint32_t i = 0; i < n_bins - 1; i++) {
                 size_t idx = (i + 1) * values.size() / n_bins;
                 bin_info_[col].splits[i] = values[idx];
             }
         }
     }

     // Now create the binned features
     binned_features_.resize(n_rows_ * n_cols_);

     for (size_t row = 0; row < n_rows_; row++) {
         for (size_t col = 0; col < n_cols_; col++) {
             float val = features_[row * n_cols_ + col];
             binned_features_[row * n_cols_ + col] = bin_info_[col].get_bin(val);
         }
     }
 }

 void DataMatrix::apply_bins(const DataMatrix& other) {
     // Copy bin information
     bin_info_ = other.bin_info();

     // Now create the binned features
     binned_features_.resize(n_rows_ * n_cols_);

     for (size_t row = 0; row < n_rows_; row++) {
         for (size_t col = 0; col < n_cols_; col++) {
             float val = features_[row * n_cols_ + col];
             binned_features_[row * n_cols_ + col] = bin_info_[col].get_bin(val);
         }
     }
 }

 } // namespace boostedpp
