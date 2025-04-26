/**
 * @file config.cpp
 * @brief Implementation of the GBDTConfig class.
 */

 #include "boostedpp/config.hpp"

 namespace boostedpp {

 bool GBDTConfig::validate() const noexcept {
     // Check basic parameters
     if (n_rounds == 0) {
         return false;
     }

     if (learning_rate <= 0.0f || learning_rate > 1.0f) {
         return false;
     }

     // Check tree parameters
     if (max_depth == 0 || max_depth > 32) {
         return false;
     }

     if (min_data_in_leaf == 0) {
         return false;
     }

     if (min_child_weight <= 0.0f) {
         return false;
     }

     if (reg_lambda < 0.0f) {
         return false;
     }

     // Check histogram parameters
     if (n_bins == 0 || n_bins > 256) {
         return false;
     }

     // Check sampling parameters
     if (subsample <= 0.0f || subsample > 1.0f) {
         return false;
     }

     if (colsample <= 0.0f || colsample > 1.0f) {
         return false;
     }

     return true;
 }

 } // namespace boostedpp
