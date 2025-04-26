/**
 * @file metrics.hpp
 * @brief Evaluation metrics for BoostedPP.
 *
 * This file contains the evaluation metrics used in BoostedPP for
 * measuring model performance.
 */

 #ifndef BOOSTEDPP_METRICS_HPP
 #define BOOSTEDPP_METRICS_HPP

 #include <vector>
 #include <string>
 #include <functional>
 #include <unordered_map>

 namespace boostedpp {

 /**
  * @brief Function type for evaluation metrics.
  */
 using MetricFunction = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

 /**
  * @brief Root Mean Squared Error metric.
  *
  * @param labels True labels.
  * @param predictions Predicted values.
  * @return RMSE value.
  */
 [[nodiscard]] float rmse(const std::vector<float>& labels, const std::vector<float>& predictions);

 /**
  * @brief Mean Absolute Error metric.
  *
  * @param labels True labels.
  * @param predictions Predicted values.
  * @return MAE value.
  */
 [[nodiscard]] float mae(const std::vector<float>& labels, const std::vector<float>& predictions);

 /**
  * @brief Binary Log Loss metric.
  *
  * @param labels True labels.
  * @param predictions Predicted values.
  * @return Log Loss value.
  */
 [[nodiscard]] float logloss(const std::vector<float>& labels, const std::vector<float>& predictions);

 /**
  * @brief Area Under ROC Curve metric.
  *
  * @param labels True labels.
  * @param predictions Predicted values.
  * @return AUC value.
  */
 [[nodiscard]] float auc(const std::vector<float>& labels, const std::vector<float>& predictions);

 /**
  * @brief Get a metric function by name.
  *
  * @param metric_name Name of the metric.
  * @return Metric function.
  */
 [[nodiscard]] MetricFunction get_metric(const std::string& metric_name);

 /**
  * @brief Get all available metric names.
  *
  * @return Vector of metric names.
  */
 [[nodiscard]] std::vector<std::string> get_available_metrics();

 } // namespace boostedpp

 #endif // BOOSTEDPP_METRICS_HPP
