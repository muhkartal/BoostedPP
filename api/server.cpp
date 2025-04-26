/**
 * @file server.cpp
 * @brief REST API server for BoostedPP.
 */

 #include <crow.h>
 #include <nlohmann/json.hpp>
 #include <string>
 #include <vector>
 #include <filesystem>
 #include <fstream>
 #include <sstream>
 #include <stdexcept>
 #include <thread>
 #include <mutex>
 #include <unordered_map>
 #include <chrono>

 #include "boostedpp/boostedpp.hpp"

 // Alias for convenience
 using json = nlohmann::json;
 namespace fs = std::filesystem;

 // Cache for loaded models
 class ModelCache {
 public:
     /**
      * @brief Get a model by name, loading it if necessary.
      */
     std::shared_ptr<boostedpp::GBDT> get_model(const std::string& model_name) {
         std::lock_guard<std::mutex> lock(mtx_);

         // Check if model is already loaded
         auto it = models_.find(model_name);
         if (it != models_.end()) {
             return it->second;
         }

         // Determine model path
         std::string models_dir = get_models_dir();
         fs::path model_path = fs::path(models_dir) / (model_name + ".json");

         if (!fs::exists(model_path)) {
             throw std::runtime_error("Model not found: " + model_name);
         }

         // Load the model
         auto model = std::make_shared<boostedpp::GBDT>();
         model->load_model(model_path.string());

         // Store in cache
         models_[model_name] = model;

         return model;
     }

     /**
      * @brief Check if a model exists.
      */
     bool model_exists(const std::string& model_name) {
         std::string models_dir = get_models_dir();
         fs::path model_path = fs::path(models_dir) / (model_name + ".json");
         return fs::exists(model_path);
     }

     /**
      * @brief List available models.
      */
     std::vector<std::string> list_models() {
         std::string models_dir = get_models_dir();
         std::vector<std::string> models;

         if (!fs::exists(models_dir)) {
             return models;
         }

         for (const auto& entry : fs::directory_iterator(models_dir)) {
             if (entry.is_regular_file() && entry.path().extension() == ".json") {
                 models.push_back(entry.path().stem().string());
             }
         }

         return models;
     }

     /**
      * @brief Clear the cache.
      */
     void clear() {
         std::lock_guard<std::mutex> lock(mtx_);
         models_.clear();
     }

 private:
     /**
      * @brief Get the models directory.
      */
     std::string get_models_dir() {
         const char* env_dir = std::getenv("MODELS_DIR");
         return env_dir ? env_dir : "./models";
     }

     std::mutex mtx_;
     std::unordered_map<std::string, std::shared_ptr<boostedpp::GBDT>> models_;
 };

 // Parse CSV data
 std::pair<std::vector<float>, std::vector<std::string>> parse_csv(const std::string& csv_data) {
     std::vector<float> features;
     std::vector<std::string> column_names;

     std::istringstream stream(csv_data);
     std::string line;

     // Parse header
     if (std::getline(stream, line)) {
         std::istringstream header_stream(line);
         std::string column;

         while (std::getline(header_stream, column, ',')) {
             column_names.push_back(column);
         }
     }

     // Parse data (one sample)
     if (std::getline(stream, line)) {
         std::istringstream data_stream(line);
         std::string value;

         while (std::getline(data_stream, value, ',')) {
             try {
                 features.push_back(std::stof(value));
             } catch (const std::exception& e) {
                 features.push_back(std::numeric_limits<float>::quiet_NaN());
             }
         }
     }

     return {features, column_names};
 }

 // Main function
 int main(int argc, char** argv) {
     // Create model cache
     ModelCache model_cache;

     // Create Crow app
     crow::SimpleApp app;

     // Set up CORS
     auto cors = crow::cors::make_cors()
         .methods("GET"_method, "POST"_method, "OPTIONS"_method)
         .prefix("/api")
         .origin("*");

     // Apply CORS middleware
     app.use(cors);

     // Get API version
     CROW_ROUTE(app, "/api/version")
     .methods("GET"_method)
     ([](const crow::request& req) {
         json response;
         response["version"] = boostedpp::version();
         response["simd"] = boostedpp::simd::get_simd_instruction_set();
         return crow::response(response.dump());
     });

     // List available models
     CROW_ROUTE(app, "/api/models")
     .methods("GET"_method)
     ([&model_cache](const crow::request& req) {
         try {
             json response;
             response["models"] = model_cache.list_models();
             return crow::response(response.dump());
         } catch (const std::exception& e) {
             return crow::response(500, e.what());
         }
     });

     // Get model information
     CROW_ROUTE(app, "/api/models/<string>")
     .methods("GET"_method)
     ([&model_cache](const crow::request& req, const std::string& model_name) {
         try {
             if (!model_cache.model_exists(model_name)) {
                 return crow::response(404, "Model not found");
             }

             auto model = model_cache.get_model(model_name);

             json response;
             response["name"] = model_name;

             const auto& config = model->config();
             response["config"]["task"] = config.task == boostedpp::Task::Binary ? "binary" : "regression";
             response["config"]["n_rounds"] = config.n_rounds;
             response["config"]["learning_rate"] = config.learning_rate;
             response["config"]["max_depth"] = config.max_depth;
             response["config"]["min_child_weight"] = config.min_child_weight;

             return crow::response(response.dump());
         } catch (const std::exception& e) {
             return crow::response(500, e.what());
         }
     });

     // Make predictions
     CROW_ROUTE(app, "/api/predict/<string>")
     .methods("POST"_method)
     ([&model_cache](const crow::request& req, const std::string& model_name) {
         try {
             if (!model_cache.model_exists(model_name)) {
                 return crow::response(404, "Model not found");
             }

             // Parse JSON request
             auto body = json::parse(req.body);

             // Handle different input formats
             std::vector<float> features;
             std::vector<std::string> feature_names;

             if (body.contains("csv")) {
                 // CSV input
                 auto [parsed_features, parsed_names] = parse_csv(body["csv"]);
                 features = parsed_features;
                 feature_names = parsed_names;
             } else if (body.contains("features") && body["features"].is_array()) {
                 // JSON array input
                 for (const auto& value : body["features"]) {
                     features.push_back(value.get<float>());
                 }
             } else {
                 return crow::response(400, "Invalid input format");
             }

             // Load model
             auto model = model_cache.get_model(model_name);

             // Create a one-row DataMatrix
             boostedpp::DataMatrix data(features, {}, 1, features.size());

             // Make prediction
             auto start_time = std::chrono::high_resolution_clock::now();
             auto predictions = model->predict(data);
             auto end_time = std::chrono::high_resolution_clock::now();

             // Calculate elapsed time
             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

             // Prepare response
             json response;
             response["prediction"] = predictions[0];
             response["model"] = model_name;
             response["time_us"] = duration.count();

             return crow::response(response.dump());
         } catch (const json::exception& e) {
             return crow::response(400, "Invalid JSON: " + std::string(e.what()));
         } catch (const std::exception& e) {
             return crow::response(500, e.what());
         }
     });

     // Determine port (default to 8080)
     uint16_t port = 8080;
     const char* env_port = std::getenv("PORT");
     if (env_port) {
         try {
             port = static_cast<uint16_t>(std::stoi(env_port));
         } catch (...) {
             // Ignore conversion errors
         }
     }

     // Print startup message
     std::cout << "BoostedPP API server starting on port " << port << std::endl;
     std::cout << "SIMD support: " << boostedpp::simd::get_simd_instruction_set() << std::endl;
     std::cout << "Available models: ";
     for (const auto& model : model_cache.list_models()) {
         std::cout << model << " ";
     }
     std::cout << std::endl;

     // Start the server
     app.port(port).multithreaded().run();

     return 0;
 }
