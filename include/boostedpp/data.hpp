/**
 * @file data.hpp
 * @brief Data structures for handling datasets in BoostedPP.
 *
 * This file contains the data structures for handling datasets in BoostedPP.
 * It defines the DataMatrix class that stores features and labels.
 */

 #ifndef BOOSTEDPP_DATA_HPP
 #define BOOSTEDPP_DATA_HPP

 #include <string>
 #include <vector>
 #include <memory>
 #include <unordered_map>
 #include <cstdint>

 namespace boostedpp {

 /**
  * @brief Special value representing a missing value.
  */
 constexpr float kMissingValue = std::numeric_limits<float>::quiet_NaN();

 /**
  * @brief Enum representing the feature bin type.
  */
 enum class BinType {
     Numerical,  ///< Numerical feature bin
     Categorical ///< Categorical feature bin
 };

 /**
  * @brief Structure for bin information.
  */
 struct BinInfo {
     BinType type;              ///< Bin type (numerical or categorical)
     std::vector<float> splits; ///< Split points for numerical features

     /**
      * @brief Get the bin index for a given value.
      * @param value The value to be binned.
      * @return The bin index.
      */
     [[nodiscard]] uint32_t get_bin(float value) const noexcept;
 };

 /**
  * @brief Class for handling datasets.
  *
  * This class is responsible for loading, preprocessing, and storing datasets.
  * It also handles feature binning and transformation.
  */
 class DataMatrix {
 public:
     /**
      * @brief Default constructor.
      */
     DataMatrix() = default;

     /**
      * @brief Constructor that loads data from a CSV file.
      * @param filename Path to the CSV file.
      * @param label_column Index of the label column (-1 means no label).
      */
     DataMatrix(const std::string& filename, int label_column);

     /**
      * @brief Constructor that takes raw data vectors.
      * @param features Feature matrix (row-major).
      * @param labels Label vector (can be empty for test data).
      * @param n_rows Number of rows.
      * @param n_cols Number of columns.
      */
     DataMatrix(const std::vector<float>& features, const std::vector<float>& labels,
                size_t n_rows, size_t n_cols);

     /**
      * @brief Create binned data for histogram-based training.
      * @param n_bins Number of bins per feature.
      */
     void create_bins(uint32_t n_bins = 256);

     /**
      * @brief Apply the binning transformation from another dataset.
      * @param other The reference dataset to derive binning from.
      */
     void apply_bins(const DataMatrix& other);

     /**
      * @brief Get the number of rows.
      * @return The number of rows.
      */
     [[nodiscard]] size_t n_rows() const noexcept { return n_rows_; }

     /**
      * @brief Get the number of columns.
      * @return The number of columns.
      */
     [[nodiscard]] size_t n_cols() const noexcept { return n_cols_; }

     /**
      * @brief Get the raw features data.
      * @return The raw features data (row-major).
      */
     [[nodiscard]] const std::vector<float>& features() const noexcept { return features_; }

     /**
      * @brief Get the binned features data.
      * @return The binned features data (row-major).
      */
     [[nodiscard]] const std::vector<uint8_t>& binned_features() const noexcept { return binned_features_; }

     /**
      * @brief Get the labels.
      * @return The labels.
      */
     [[nodiscard]] const std::vector<float>& labels() const noexcept { return labels_; }

     /**
      * @brief Get the bin information.
      * @return The bin information for each feature.
      */
     [[nodiscard]] const std::vector<BinInfo>& bin_info() const noexcept { return bin_info_; }

     /**
      * @brief Get a feature value.
      * @param row Row index.
      * @param col Column index.
      * @return The feature value.
      */
     [[nodiscard]] float get_feature(size_t row, size_t col) const {
         return features_[row * n_cols_ + col];
     }

     /**
      * @brief Get a binned feature value.
      * @param row Row index.
      * @param col Column index.
      * @return The binned feature value.
      */
     [[nodiscard]] uint8_t get_binned_feature(size_t row, size_t col) const {
         return binned_features_[row * n_cols_ + col];
     }

     /**
      * @brief Get a label.
      * @param row Row index.
      * @return The label.
      */
     [[nodiscard]] float get_label(size_t row) const {
         return labels_[row];
     }

 private:
     size_t n_rows_ = 0;                   ///< Number of rows
     size_t n_cols_ = 0;                   ///< Number of columns
     std::vector<float> features_;         ///< Raw features (row-major)
     std::vector<uint8_t> binned_features_; ///< Binned features (row-major)
     std::vector<float> labels_;           ///< Labels
     std::vector<BinInfo> bin_info_;       ///< Bin information for each feature

     /**
      * @brief Load data from a CSV file.
      * @param filename Path to the CSV file.
      * @param label_column Index of the label column (-1 means no label).
      */
     void load_csv(const std::string& filename, int label_column);
 };

 } // namespace boostedpp

 #endif // BOOSTEDPP_DATA_HPP
