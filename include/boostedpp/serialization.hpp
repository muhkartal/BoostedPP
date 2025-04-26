/**
 * @file serialization.hpp
 * @brief Model serialization utilities for BoostedPP.
 *
 * This file contains utilities for serializing and deserializing models
 * to and from various formats, including XGBoost compatibility.
 */

 #ifndef BOOSTEDPP_SERIALIZATION_HPP
 #define BOOSTEDPP_SERIALIZATION_HPP

 #include <string>
 #include <vector>
 #include <nlohmann/json.hpp>

 #include "tree.hpp"
 #include "gbdt.hpp"

 namespace boostedpp {

 /**
  * @brief Save a GBDT model to a JSON file.
  *
  * @param model GBDT model to save.
  * @param filename Path to the output file.
  */
 void save_model_to_json(const GBDT& model, const std::string& filename);

 /**
  * @brief Load a GBDT model from a JSON file.
  *
  * @param filename Path to the input file.
  * @return Loaded GBDT model.
  */
 GBDT load_model_from_json(const std::string& filename);

 /**
  * @brief Convert a GBDT model to XGBoost JSON format.
  *
  * @param model GBDT model to convert.
  * @return JSON representation of the model in XGBoost format.
  */
 nlohmann::json convert_to_xgboost_json(const GBDT& model);

 /**
  * @brief Convert a model from XGBoost JSON format to GBDT.
  *
  * @param json JSON representation of the model in XGBoost format.
  * @return Converted GBDT model.
  */
 GBDT convert_from_xgboost_json(const nlohmann::json& json);

 /**
  * @brief Save a model to a file in XGBoost JSON format.
  *
  * @param model GBDT model to save.
  * @param filename Path to the output file.
  */
 void save_model_to_xgboost_json(const GBDT& model, const std::string& filename);

 /**
  * @brief Load a model from a file in XGBoost JSON format.
  *
  * @param filename Path to the input file.
  * @return Loaded GBDT model.
  */
 GBDT load_model_from_xgboost_json(const std::string& filename);

 } // namespace boostedpp

 #endif // BOOSTEDPP_SERIALIZATION_HPP
