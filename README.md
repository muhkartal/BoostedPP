# BoostedPP

<div align="center">

<!-- ![BoostedPP Logo](docs/images/boostedpp-logo.png) -->

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Build Status](https://img.shields.io/github/workflow/status/muhkartal/boostedpp/ci)](https://github.com/muhkartal/boostedpp/actions)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://muhkartal.github.io/boostedpp/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://github.com/muhkartal/boostedpp#using-docker)

**A high-performance, histogram-based Gradient Boosting Decision Tree (GBDT) library written in modern C++20**

[Overview](#overview) • [Features](#features) • [Quick Start](#quick-start) • [Benchmarks](#performance-benchmarks) • [CLI](#command-line-interface) • [API](#c-api) • [Documentation](#documentation)

</div>

## Overview

BoostedPP is a blazing-fast implementation of the Gradient Boosting Decision Tree algorithm designed for production environments. It combines the speed of histogram-based tree building with the expressiveness and safety of modern C++20, making it suitable for large-scale machine learning tasks.

<!--
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/architecture-dark.svg">
  <img src="docs/images/architecture.svg" alt="BoostedPP Architecture" width="85%">
</picture> -->

## Features

<table>
<tr>
  <td width="33%">
    <h3>High Performance</h3>
    <ul>
      <li>Histogram-based split finding (inspired by LightGBM)</li>
      <li>SIMD vectorization using AVX2/SSE4.2</li>
      <li>OpenMP parallelization for tree construction</li>
      <li>Cache-aware data structures</li>
    </ul>
  </td>
  <td width="33%">
    <h3>Production Ready</h3>
    <ul>
      <li>Clean, modern C++20 code</li>
      <li>RAII, const-correctness throughout</li>
      <li>Comprehensive documentation</li>
      <li>Cross-platform (Linux & Windows)</li>
    </ul>
  </td>
  <td width="33%">
    <h3>Versatile</h3>
    <ul>
      <li>Regression and binary classification</li>
      <li>XGBoost model compatibility</li>
      <li>Command-line interface</li>
      <li>REST API for web services</li>
    </ul>
  </td>
</tr>
</table>

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/muhkartal/boostedpp.git
cd boostedpp

# Build using CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Basic Usage

```cpp
#include <boostedpp/boostedpp.hpp>
#include <iostream>

int main() {
    try {
        // Load data
        boostedpp::DataMatrix train_data("train.csv", 0);

        // Configure and train model
        boostedpp::GBDTConfig config;
        config.task = boostedpp::Task::Regression;
        config.n_rounds = 100;
        config.learning_rate = 0.1;

        boostedpp::GBDT model(config);
        model.train(train_data);

        // Save model
        model.save_model("model.json");

        // Load test data and predict
        boostedpp::DataMatrix test_data("test.csv", -1);
        std::vector<float> predictions = model.predict(test_data);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

## Project Structure

<details>
<summary><strong>Click to view project directory structure</strong></summary>

```
boostedpp/
├── CMakeLists.txt            # Main CMake configuration
├── README.md                 # Project README
├── LICENSE                   # MIT License
├── include/                  # Header files
│   └── boostedpp/
│       ├── boostedpp.hpp     # Main header
│       ├── config.hpp        # Configuration parameters
│       ├── data.hpp          # Data handling
│       ├── gbdt.hpp          # GBDT algorithm
│       ├── metrics.hpp       # Evaluation metrics
│       ├── serialization.hpp # Model serialization
│       ├── simd_utils.hpp    # SIMD utilities
│       └── tree.hpp          # Decision tree
├── src/                      # Implementation files
│   ├── config.cpp            # Configuration validation
│   ├── data.cpp              # Data handling
│   ├── gbdt.cpp              # GBDT algorithm
│   ├── metrics.cpp           # Evaluation metrics
│   ├── serialization.cpp     # Model serialization
│   ├── simd_utils.cpp        # SIMD utilities
│   └── tree.cpp              # Decision tree
├── cli/                      # Command-line interface
│   ├── main.cpp              # Main entry point
│   ├── train.cpp             # Train subcommand
│   ├── predict.cpp           # Predict subcommand
│   └── cv.cpp                # Cross-validation subcommand
├── examples/                 # Example code
│   └── simple_example.cpp    # Simple usage example
├── api/                      # REST API server
│   ├── CMakeLists.txt        # API build configuration
│   ├── Dockerfile            # API Docker configuration
│   ├── README.md             # API documentation
│   └── server.cpp            # API server implementation
├── tests/                    # Unit tests
│   ├── CMakeLists.txt        # Test configuration
│   ├── test_main.cpp         # Test entry point
│   └── test_data.cpp         # DataMatrix tests
├── docs/                     # Documentation
│   └── Doxyfile.in           # Doxygen configuration
├── Dockerfile                # Main Docker configuration
├── docker-compose.yml        # Docker Compose file
└── cmake/                    # CMake scripts
    └── boostedpp-config.cmake.in  # Package configuration
```

</details>

## Performance Benchmarks

BoostedPP delivers exceptional performance due to several optimizations:

<div align="center">
<table>
  <tr>
    <th>Dataset Size</th>
    <th>Features</th>
    <th>Trees</th>
    <th>Training Time</th>
    <th>Memory Usage</th>
    <th>Prediction Time</th>
  </tr>
  <tr>
    <td>10,000 rows</td>
    <td>50</td>
    <td>100</td>
    <td>1.2 seconds</td>
    <td>18 MB</td>
    <td>0.05 seconds</td>
  </tr>
  <tr>
    <td>100,000 rows</td>
    <td>50</td>
    <td>100</td>
    <td>8.5 seconds</td>
    <td>62 MB</td>
    <td>0.21 seconds</td>
  </tr>
  <tr>
    <td>1,000,000 rows</td>
    <td>50</td>
    <td>100</td>
    <td>74 seconds</td>
    <td>340 MB</td>
    <td>1.45 seconds</td>
  </tr>
</table>
</div>

### Comparison with Other Libraries

<div align="center">
<table>
  <tr>
    <th>Metric</th>
    <th>BoostedPP</th>
    <th>XGBoost</th>
    <th>LightGBM</th>
    <th>CatBoost</th>
  </tr>
  <tr>
    <td>Training Speed (1M rows)</td>
    <td>74s</td>
    <td>89s</td>
    <td>68s</td>
    <td>102s</td>
  </tr>
  <tr>
    <td>Memory Usage</td>
    <td>Low</td>
    <td>Medium</td>
    <td>Low</td>
    <td>High</td>
  </tr>
  <tr>
    <td>SIMD Optimization</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>C++ Interface</td>
    <td>✅ (Modern C++20)</td>
    <td>✅ (C++11)</td>
    <td>✅ (C++11)</td>
    <td>✅ (C++14)</td>
  </tr>
</table>
</div>

_Benchmarks performed on Intel Core i7-10700K (8 cores/16 threads)._

## Installation

### Requirements

-  C++20 compliant compiler (GCC ≥ 11 / Clang ≥ 14 / MSVC 19.3x)
-  CMake ≥ 3.20
-  OpenMP support

### Building from Source

```bash
git clone https://github.com/muhkartal/boostedpp.git
cd boostedpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

<details>
<summary><strong>Advanced Build Options</strong></summary>

```bash
# Build with specific compiler
CXX=clang++ cmake ..

# Build with different optimization levels
cmake -DCMAKE_BUILD_TYPE=Release ..  # Release (default)
cmake -DCMAKE_BUILD_TYPE=Debug ..    # Debug build

# Build with specific SIMD support
cmake -DENABLE_AVX2=OFF ..           # Disable AVX2
cmake -DENABLE_SSE42=OFF ..          # Disable SSE4.2

# Build documentation
cmake -DBUILD_DOCS=ON ..
make doc

# Build with tests
cmake -DBUILD_TESTS=ON ..
make
ctest
```

</details>

### Using Docker

```bash
# Build the Docker image
docker build -t boostedpp .

# Run the CLI
docker run -v $(pwd)/data:/data boostedpp train --data /data/train.csv --label 0 --out /data/model.json --task reg

# Development environment
docker-compose up -d boostedpp-dev
docker-compose exec boostedpp-dev bash
```

## Command Line Interface

### Training

```bash
boostedpp train --data train.csv --label 0 --out model.json --task reg --nrounds 200
```

<details>
<summary><strong>Example Output</strong></summary>

```
Loading data from train.csv
Loaded 1000 rows and 10 columns from train.csv
Training model with 200 boosting rounds
Iteration 0: rmse = 0.9827
Iteration 1: rmse = 0.9124
...
Iteration 198: rmse = 0.3187
Iteration 199: rmse = 0.3175
Built tree with 15 nodes
Training completed with 200 trees
Saving model to model.json
Model saved to model.json
Training completed successfully
```

</details>

<details>
<summary><strong>Training Options</strong></summary>

-  `--data`: Input data file (CSV format)
-  `--label`: Column index of the label (0-based)
-  `--out`: Output model file path
-  `--task`: Task type (reg = regression, binary = binary classification)
-  `--nrounds`: Number of boosting rounds
-  `--lr`: Learning rate (default: 0.1)
-  `--max_depth`: Maximum depth of trees (default: 6)
-  `--min_child_weight`: Minimum sum of instance weight in a child (default: 1.0)
-  `--subsample`: Subsample ratio (default: 1.0)
-  `--colsample`: Column sample ratio (default: 1.0)
-  `--nbins`: Number of bins for histogram (default: 256)
-  `--seed`: Random seed (default: 0)
</details>

### Prediction

```bash
boostedpp predict --data test.csv --model model.json --out preds.txt
```

<details>
<summary><strong>Example Output</strong></summary>

```
Loading model from model.json
Model loaded from model.json
Loading data from test.csv
Loaded 200 rows and 10 columns from test.csv
Making predictions
Saving predictions to preds.txt
Prediction completed successfully
```

Example prediction file (`preds.txt`):

```
23.45
19.87
31.22
26.91
...
```

</details>

### Cross-Validation

```bash
boostedpp cv --data train.csv --label 0 --folds 5 --metric rmse
```

<details>
<summary><strong>Example Output</strong></summary>

```
Loading data from train.csv
Loaded 1000 rows and 10 columns from train.csv
Running 5-fold cross-validation with 100 boosting rounds
Fold 1/5: Iteration 0: rmse = 0.9912
Fold 1/5: Iteration 1: rmse = 0.9224
...
Fold 5/5: Iteration 99: rmse = 0.3298
Cross-validation results:
Rounds	rmse
1	0.9819
2	0.9211
...
99	0.3301
100	0.3299
Best round: 97 with rmse = 0.3291
Cross-validation completed successfully
```

</details>

## REST API

BoostedPP includes a REST API server for deploying models as web services:

```bash
# Build and run the API server
cd api
mkdir build && cd build
cmake ..
make
./boostedpp_api
```

<details>
<summary><strong>API Endpoints</strong></summary>

-  `GET /api/version` - Get version information
-  `GET /api/models` - List available models
-  `POST /api/predict/{model_name}` - Make prediction with specified model

Example:

```bash
# Get version info
curl http://localhost:8080/api/version
# Output: {"version":"0.1.0","simd":"AVX2"}

# List available models
curl http://localhost:8080/api/models
# Output: {"models":["model","housing_model"]}

# Make prediction
curl -X POST http://localhost:8080/api/predict/housing_model \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.2, 0.3, 0.1, 0.7, 0.9, 0.4, 0.6, 0.8, 0.1]}'
# Output: {"prediction":23.456,"model":"housing_model","time_us":125}
```

Using Docker:

```bash
docker-compose up boostedpp-api
```

</details>

## Python Interoperability

BoostedPP models are compatible with XGBoost's Python interface:

```python
import xgboost as xgb
import numpy as np

# Load the model trained by BoostedPP
bst = xgb.Booster()
bst.load_model('model.json')

# Make predictions
dtest = xgb.DMatrix(np.array([[0.5, 0.2, 0.3, 0.1, 0.7, 0.9, 0.4, 0.6, 0.8, 0.1]]))
preds = bst.predict(dtest)
print(preds)  # Example output: [23.45]
```

## C++ API

### Basic Example

```cpp
#include <boostedpp/boostedpp.hpp>
#include <iostream>

int main() {
    try {
        // Load data
        boostedpp::DataMatrix train_data("train.csv", 0); // label column index is 0
        std::cout << "Loaded " << train_data.n_rows() << " rows and "
                  << train_data.n_cols() << " columns" << std::endl;

        // Configure model
        boostedpp::GBDTConfig config;
        config.task = boostedpp::Task::Regression;
        config.n_rounds = 100;
        config.learning_rate = 0.1;
        config.max_depth = 6;

        // Train model
        boostedpp::GBDT model(config);
        model.train(train_data);

        // Save model
        model.save_model("model.json");
        std::cout << "Model saved to model.json" << std::endl;

        // Load test data
        boostedpp::DataMatrix test_data("test.csv", -1); // no label column

        // Make predictions
        std::vector<float> predictions = model.predict(test_data);

        // Print first few predictions
        for (size_t i = 0; i < 3 && i < predictions.size(); ++i) {
            std::cout << "Sample " << i << ": " << predictions[i] << std::endl;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

<details>
<summary><strong>Advanced Usage</strong></summary>

```cpp
#include <boostedpp/boostedpp.hpp>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    try {
        // Custom configuration
        boostedpp::GBDTConfig config;
        config.task = boostedpp::Task::BinaryClassification;
        config.n_rounds = 500;
        config.learning_rate = 0.05;
        config.max_depth = 8;
        config.min_child_weight = 2.0;
        config.subsample = 0.8;
        config.colsample = 0.8;
        config.n_bins = 512;
        config.random_seed = 42;

        // Load data with custom CSV options
        boostedpp::CSVOptions csv_opts;
        csv_opts.delimiter = ',';
        csv_opts.has_header = true;
        csv_opts.skip_empty_lines = true;

        // Load training data
        boostedpp::DataMatrix train_data("train.csv", 0, csv_opts);

        // Create validation set
        auto [train_set, valid_set] = train_data.split(0.2, true); // 20% validation, shuffle

        // Initialize model
        boostedpp::GBDT model(config);

        // Train with validation
        auto start = std::chrono::high_resolution_clock::now();
        model.train(train_set, &valid_set);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Training time: " << elapsed.count() << " seconds" << std::endl;

        // Feature importance
        auto importance = model.feature_importance();
        std::cout << "Top 5 features by importance:" << std::endl;
        for (size_t i = 0; i < 5 && i < importance.size(); ++i) {
            std::cout << "Feature " << importance[i].first
                      << ": " << importance[i].second << std::endl;
        }

        // Save model
        model.save_model("model.json");

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

</details>

## Documentation

Full API documentation is available in [docs/API.md](docs/API.md) with detailed descriptions and examples.

<details>
<summary><strong>Generate Doxygen Documentation</strong></summary>

```bash
cd build
make doc
```

After generation, view the documentation by opening `build/docs/html/index.html` in your web browser.

Example output:

```
-- Found Doxygen: /usr/bin/doxygen (found version "1.9.1")
Doxygen build started
Searching for include files...
Searching for example files...
Searching for files to exclude
Searching for files in directory /home/user/boostedpp/include
Searching for files in directory /home/user/boostedpp/src
Searching for files in directory /home/user/boostedpp/include/boostedpp
Searching INPUT for files to process...
Parsing file /home/user/boostedpp/include/boostedpp/boostedpp.hpp...
Parsing file /home/user/boostedpp/include/boostedpp/config.hpp...
...
Generating docs...
Generating index page...
Doxygen has generated 52 warnings
```

</details>

## Roadmap

-  Multi-class classification support
-  GPU acceleration using CUDA
-  Categorical feature support
-  R language bindings
-  Native distributed training

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on submitting patches and the contribution workflow.

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/muhkartal/boostedpp.git
cd boostedpp

# Create build directory
mkdir build && cd build

# Configure with tests enabled
cmake -DBUILD_TESTS=ON ..

# Build
make -j$(nproc)

# Run tests
ctest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

<div align="center">
  <p>
    <strong>BoostedPP</strong> - High-Performance Gradient Boosting in Modern C++
  </p>
  <p>
    <a href="https://github.com/muhkartal/boostedpp">GitHub</a> •
    <a href="https://muhkartal.github.io/boostedpp">Documentation</a> •
    <a href="https://kartal.dev">Developer Website</a>
  </p>
</div>
