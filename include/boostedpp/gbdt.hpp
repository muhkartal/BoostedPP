/**
 * @file gbdt.hpp
 * @brief Gradient Boosting Decision Tree algorithm implementation.
 *
 * This file contains the implementation of the Gradient Boosting Decision Tree algorithm.
 * It defines the GBDT class that manages the training and prediction process.
 */

 #ifndef BOOSTEDPP_GBDT_HPP
 #define BOOSTEDPP_GBDT_HPP

 #include <vector>
 #include <memory>
 #include <string>
 #include <nlohmann/json.hpp>

 #include "config.hpp"
 #include "data.hpp"
 #include "tree.hpp"

 namespace boostedpp {

 /**
  * @brief Class implementing the Gradient Boosting Decision Tree algorithm.
  *
  * This class manages the training and prediction process of the GBDT algorithm.
  * It builds an ensemble of decision trees to minimize the loss function.
  */
 class GBDT {
 public:
     /**
      * @brief Default constructor.
      */
     GBDT() = default;

     /**
      * @brief Constructor with configuration.
      * @param config GBDT configuration.
      */
     explicit GBDT(const GBDTConfig& config);

     /**
      * @brief Train the model on the given dataset.
      * @param data Training data.
      */
     void train(const DataMatrix& data);

     /**
      * @brief Predict the output for a dataset.
      * @param data Test data.
      * @return Vector of predictions.
      */
     [[nodiscard]] std::vector<float> predict(const DataMatrix& data) const;

     /**
      * @brief Run cross-validation.
      * @param data Training data.
      * @param n_folds Number of folds.
      * @return Vector of evaluation results for each round.
      */
     [[nodiscard]] std::vector<float> cv(const DataMatrix& data, uint32_t n_folds) const;

     /**
      * @brief Save the model to a file.
      * @param filename Path to the output file.
      */
     void save_model(const std::string& filename) const;

     /**
      * @brief Load the model from a file.
      * @param filename Path to the input file.
      */
     void load_model(const std::string& filename);

     /**
      * @brief Convert the model to XGBoost JSON format.
      * @return JSON representation of the model.
      */
     [[nodiscard]] nlohmann::json to_xgboost_json() const;

     /**
      * @brief Load a model from XGBoost JSON format.
      * @param json JSON representation of the model.
      */
     void from_xgboost_json(const nlohmann::json& json);

     /**
      * @brief Get the trees in the model.
      * @return Vector of trees.
      */
     [[nodiscard]] const std::vector<Tree>& trees() const noexcept { return trees_; }

     /**
      * @brief Get the configuration.
      * @return GBDT configuration.
      */
     [[nodiscard]] const GBDTConfig& config() const noexcept { return config_; }

 private:
     GBDTConfig config_;             ///< GBDT configuration
     std::vector<Tree> trees_;       ///< Vector of trees
     float base_score_ = 0.0f;       ///< Base prediction score

     /**
      * @brief Initialize gradients and hessians for the first iteration.
      * @param data Training data.
      * @param out_gradients Output gradients.
      * @param out_hessians Output hessians.
      */
     void init_gradients(
         const DataMatrix& data,
         std::vector<float>& out_gradients,
         std::vector<float>& out_hessians) const;

     /**
      * @brief Update gradients and hessians after each iteration.
      * @param data Training data.
      * @param predictions Current predictions.
      * @param tree Latest tree.
      * @param out_gradients Output gradients.
      * @param out_hessians Output hessians.
      */
     void update_gradients(
         const DataMatrix& data,
         std::vector<float>& predictions,
         const Tree& tree,
         std::vector<float>& out_gradients,
         std::vector<float>& out_hessians) const;

     /**
      * @brief Calculate the initial base score.
      * @param data Training data.
      * @return Base score.
      */
     [[nodiscard]] float calculate_base_score(const DataMatrix& data) const;

     /**
      * @brief Evaluate the model on a dataset.
      * @param data Test data.
      * @param predictions Predictions.
      * @return Evaluation metric value.
      */
     [[nodiscard]] float evaluate(
         const DataMatrix& data,
         const std::vector<float>& predictions) const;
 };

 } // namespace boostedpp

 #endif // BOOSTEDPP_GBDT_HPP
