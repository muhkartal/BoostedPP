/**
 * @file test_data.cpp
 * @brief Tests for the DataMatrix class.
 */

 #include <gtest/gtest.h>
 #include "boostedpp/data.hpp"
 #include <vector>
 #include <fstream>
 #include <cmath>

 // Create a temporary CSV file for testing
 std::string create_test_csv() {
     std::string filename = "test_data.csv";
     std::ofstream file(filename);
     file << "feature1,feature2,label\n";
     file << "1.0,2.0,0.0\n";
     file << "2.0,3.0,1.0\n";
     file << "3.0,4.0,0.0\n";
     file << "4.0,5.0,1.0\n";
     file << "5.0,6.0,0.0\n";
     file.close();
     return filename;
 }

 // Test constructing DataMatrix from raw data
 TEST(DataMatrixTest, ConstructFromRawData) {
     std::vector<float> features = {1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f};
     std::vector<float> labels = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

     boostedpp::DataMatrix data(features, labels, 5, 2);

     EXPECT_EQ(data.n_rows(), 5);
     EXPECT_EQ(data.n_cols(), 2);
     EXPECT_EQ(data.get_feature(0, 0), 1.0f);
     EXPECT_EQ(data.get_feature(0, 1), 2.0f);
     EXPECT_EQ(data.get_feature(1, 0), 2.0f);
     EXPECT_EQ(data.get_label(0), 0.0f);
     EXPECT_EQ(data.get_label(1), 1.0f);
 }

 // Test loading DataMatrix from CSV
 TEST(DataMatrixTest, LoadFromCSV) {
     std::string filename = create_test_csv();

     boostedpp::DataMatrix data(filename, 2); // Label column is 2 (0-indexed)

     EXPECT_EQ(data.n_rows(), 5);
     EXPECT_EQ(data.n_cols(), 2);
     EXPECT_EQ(data.get_feature(0, 0), 1.0f);
     EXPECT_EQ(data.get_feature(0, 1), 2.0f);
     EXPECT_EQ(data.get_feature(1, 0), 2.0f);
     EXPECT_EQ(data.get_label(0), 0.0f);
     EXPECT_EQ(data.get_label(1), 1.0f);

     // Clean up
     std::remove(filename.c_str());
 }

 // Test binning
 TEST(DataMatrixTest, Binning) {
     std::vector<float> features = {1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f};
     std::vector<float> labels = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

     boostedpp::DataMatrix data(features, labels, 5, 2);

     // Create bins
     data.create_bins(4); // 4 bins

     // Check that binned features were created
     EXPECT_EQ(data.binned_features().size(), 10);

     // Check binning consistency
     for (size_t i = 0; i < 5; i++) {
         for (size_t j = 0; j < 2; j++) {
             uint8_t bin = data.get_binned_feature(i, j);
             EXPECT_GE(bin, 0);
             EXPECT_LT(bin, 4);
         }
     }

     // Check that the bin information is created
     EXPECT_EQ(data.bin_info().size(), 2);
 }

 // Test handling missing values
 TEST(DataMatrixTest, MissingValues) {
     std::vector<float> features = {1.0f, 2.0f,
                                    2.0f, boostedpp::kMissingValue,
                                    3.0f, 4.0f,
                                    boostedpp::kMissingValue, 5.0f,
                                    5.0f, 6.0f};
     std::vector<float> labels = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

     boostedpp::DataMatrix data(features, labels, 5, 2);

     // Create bins
     data.create_bins(4); // 4 bins

     // Missing values should be in the last bin
     EXPECT_EQ(data.get_binned_feature(1, 1), 3); // missing value in bin 3 (0-indexed)
     EXPECT_EQ(data.get_binned_feature(3, 0), 3); // missing value in bin 3 (0-indexed)
 }

 // Test applying bins from another dataset
 TEST(DataMatrixTest, ApplyBins) {
     std::vector<float> train_features = {1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f};
     std::vector<float> train_labels = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

     boostedpp::DataMatrix train_data(train_features, train_labels, 5, 2);
     train_data.create_bins(4); // 4 bins

     std::vector<float> test_features = {1.5f, 2.5f, 2.5f, 3.5f, 3.5f, 4.5f};
     std::vector<float> test_labels; // No labels

     boostedpp::DataMatrix test_data(test_features, test_labels, 3, 2);
     test_data.apply_bins(train_data);

     // Check that binned features were created
     EXPECT_EQ(test_data.binned_features().size(), 6);

     // Check that bin information is copied
     EXPECT_EQ(test_data.bin_info().size(), 2);
 }
