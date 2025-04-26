# BoostedPP C++ API Reference

This document provides detailed information about the BoostedPP C++ API.

## Core Classes

### `GBDTConfig`

Configuration parameters for the GBDT algorithm.

```cpp
struct GBDTConfig {
    // Basic parameters
    Task task = Task::Regression;   // Task type (regression or binary classification)
    uint32_t n_rounds = 100;        // Number of boosting rounds
    float learning_rate = 0.1f;     // Learning rate

    // Tree parameters
    uint32_t max_depth = 6;         // Maximum depth of trees
    uint32_t min_data_in_leaf = 20; // Minimum number of instances in a leaf
    float min_child_weight = 1.0f;  // Minimum sum of instance weight in a child
    float reg_lambda = 1.0f;        // L2 regularization

    // Histogram parameters
    uint32_t n_bins = 256;          // Number of bins for histogram

    // Sampling parameters
    float subsample = 1.0f;         // Subsample ratio
    float colsample = 1.0f;         // Column sample ratio
    uint32_t seed = 0;              // Random seed

    // Parallelization
    int n_threads = -1;             // Number of threads (-1 means using all available)

    // Metrics
    std::string metric = "rmse";    // Evaluation metric

    // Validation
    bool validate() const noexcept;
};
```

### `DataMatrix`

Class for handling datasets.

```cpp
class DataMatrix {
public:
    // Constructors
    DataMatrix();
    DataMatrix(const std::string& filename, int label_column);
    DataMatrix(const std::vector<float>& features, const std::vector<float>& labels,
               size_t n_rows, size_t n_cols);

    // Binning methods
    void create_bins(uint32_t n_bins = 256);
    void apply_bins(const DataMatrix& other);

    // Accessors
    size_t n_rows() const noexcept;
    size_t n_cols() const noexcept;
    const std::vector<float>& features() const noexcept;
    const std::vector<uint8_t>& binned_features() const noexcept;
    const std::vector<float>& labels() const noexcept;
    const std::vector<BinInfo>& bin_info() const noexcept;

    // Element access
    float get_feature(size_t row, size_t col) const;
    uint8_t get_binned_feature(size_t row, size_t col) const;
    float get_label(size_t row) const;
};
```

### `Tree`

Class representing a decision tree.

```cpp
class Tree {
public:
    // Constructors
    Tree();
    explicit Tree(const GBDTConfig& config);

    // Training and prediction
    void build(const DataMatrix& data,
               const std::vector<float>& gradients,
               const std::vector<float>& hessians,
               const std::vector<uint32_t>& row_indices);
    float predict_one(const std::vector<float>& features) const;
    void predict(const DataMatrix& data, std::vector<float>& out_predictions) const;

    // Serialization
    nlohmann::json to_xgboost_json() const;
    void from_xgboost_json(const nlohmann::json& json);

    // Accessors
    size_t size() const noexcept;
    const std::vector<TreeNode>& nodes() const noexcept;
};
```

### `GBDT`

Class implementing the Gradient Boosting Decision Tree algorithm.

```cpp
class GBDT {
public:
    // Constructors
    GBDT();
    explicit GBDT(const GBDTConfig& config);

    // Training and prediction
    void train(const DataMatrix& data);
    std::vector<float> predict(const DataMatrix& data) const;
    std::vector<float> cv(const DataMatrix& data, uint32_t n_folds) const;

    // Model I/O
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);

    // XGBoost compatibility
    nlohmann::json to_xgboost_json() const;
    void from_xgboost_json(const nlohmann::json& json);

    // Accessors
    const std::vector<Tree>& trees() const noexcept;
    const GBDTConfig& config() const noexcept;
};
```

## Usage Examples

### Basic Training and Prediction

```cpp
#include <boostedpp/boostedpp.hpp>
#include <iostream>

int main() {
    try {
        // Load data
        boostedpp::DataMatrix train_data("train.csv", 0); // 0 is the label column

        // Configure model
        boostedpp::GBDTConfig config;
        config.task = boostedpp::Task::Regression;
        config.n_rounds = 100;
        config.learning_rate = 0.1f;
        config.max_depth = 6;

        // Train model
        boostedpp::GBDT model(config);
        model.train(train_data);

        // Save model
        model.save_model("model.json");

        // Load test data
        boostedpp::DataMatrix test_data("test.csv", -1); // No label column

        // Make predictions
        std::vector<float> predictions = model.predict(test_data);

        // Print predictions
        for (size_t i = 0; i < std::min(5UL, predictions.size()); i++) {
            std::cout << "Sample " << i << ": " << predictions[i] << std::endl;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

### Binary Classification

```cpp
#include <boostedpp/boostedpp.hpp>
#include <iostream>

int main() {
    try {
        // Load data
        boostedpp::DataMatrix train_data("binary_train.csv", 0);

        // Configure model for binary classification
        boostedpp::GBDTConfig config;
        config.task = boostedpp::Task::Binary;
        config.n_rounds = 50;
        config.learning_rate = 0.1f;
        config.metric = "logloss";

        // Train model
        boostedpp::GBDT model(config);
        model.train(train_data);

        // Cross-validation
        std::vector<float> cv_results = model.cv(train_data, 5);
        std::cout << "Cross-validation logloss: "
                  << cv_results[cv_results.size() - 1] << std::endl;

        // Save model
        model.save_model("binary_model.json");

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

### XGBoost Compatibility

```cpp
#include <boostedpp/boostedpp.hpp>
#include <iostream>

int main() {
    try {
        // Train a model
        boostedpp::DataMatrix train_data("train.csv", 0);

        boostedpp::GBDTConfig config;
        config.task = boostedpp::Task::Regression;
        config.n_rounds = 100;

        boostedpp::GBDT model(config);
        model.train(train_data);

        // Save in XGBoost format
        model.save_model_to_xgboost_json("model_xgb.json");
        std::cout << "Model saved in XGBoost format" << std::endl;

        // Load from XGBoost format
        boostedpp::GBDT loaded_model = boostedpp::load_model_from_xgboost_json("model_xgb.json");
        std::cout << "Model loaded from XGBoost format" << std::endl;

        // Make predictions with loaded model
        std::vector<float> predictions = loaded_model.predict(train_data);
        std::cout << "Made predictions with loaded model" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

### Multi-threading Control

```cpp
#include <boostedpp/boostedpp.hpp>
#include <iostream>

int main() {
    try {
        // Load data
        boostedpp::DataMatrix train_data("train.csv", 0);

        // Configure model with specific thread count
        boostedpp::GBDTConfig config;
        config.task = boostedpp::Task::Regression;
        config.n_rounds = 100;
        config.n_threads = 4; // Use 4 threads

        // Train model
        boostedpp::GBDT model(config);
        model.train(train_data);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

## Error Handling

BoostedPP uses exceptions for error handling. The main exceptions to be aware of are:

-  `std::invalid_argument`: Thrown when input parameters are invalid.
-  `std::runtime_error`: Thrown when an operation fails at runtime.
-  `std::out_of_range`: Thrown when accessing elements out of bounds.

It's recommended to wrap BoostedPP code in try-catch blocks to handle these exceptions properly.

## Thread Safety

-  The `GBDT` class is not thread-safe for concurrent modification.
-  Multiple threads can safely call const methods like `predict()` on the same `GBDT` instance.
-  Each thread should use its own `DataMatrix` instances.

## Performance Tips

1. **Data Preprocessing**:

   -  Ensure features are properly scaled for best results.
   -  Consider removing or imputing missing values before training.

2. **Parameter Tuning**:

   -  Use cross-validation (`cv()`) to find optimal parameters.
   -  Start with a small number of rounds and increase gradually.

3. **Memory Usage**:

   -  The binned representation uses much less memory than raw features.
   -  For very large datasets, consider training on subsampled data.

4. **Prediction Speed**:
   -  Consider using a smaller number of trees for inference if speed is critical.
   -  Use appropriate thread count based on your hardware.
