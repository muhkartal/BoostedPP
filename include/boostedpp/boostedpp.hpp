/**
 * @file boostedpp.hpp
 * @brief Main header file for the BoostedPP library.
 *
 * This is the main header file for the BoostedPP library, which provides
 * a high-performance Gradient Boosting Decision Tree implementation.
 *
 * @author BoostedPP Team
 * @version 0.1.0
 * @date 2025-04-26
 */

 #ifndef BOOSTEDPP_HPP
 #define BOOSTEDPP_HPP

 #include "config.hpp"
 #include "data.hpp"
 #include "tree.hpp"
 #include "gbdt.hpp"
 #include "metrics.hpp"
 #include "serialization.hpp"

 /**
  * @namespace boostedpp
  * @brief Namespace for the BoostedPP library.
  */
 namespace boostedpp {

 /**
  * @brief Get the version of the BoostedPP library.
  * @return String containing the version number.
  */
 inline std::string version() {
     return "0.1.0";
 }

 /**
  * @brief Get information about the BoostedPP library.
  * @return String containing information about the library.
  */
 inline std::string info() {
     std::string info_str = "BoostedPP - Gradient Boosting Decision Tree Library\n";
     info_str += "Version: " + version() + "\n";
     info_str += "SIMD Support: " + std::string(simd::get_simd_instruction_set()) + "\n";
     return info_str;
 }

 } // namespace boostedpp

 #endif // BOOSTEDPP_HPP
