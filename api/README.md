# BoostedPP REST API

This directory contains a REST API server for BoostedPP, allowing prediction using trained models over HTTP.

## API Endpoints

### `GET /api/version`

Get the version information of the BoostedPP library.

**Example Response:**

```json
{
   "version": "0.1.0",
   "simd": "AVX2"
}
```

### `GET /api/models`

List all available models.

**Example Response:**

```json
{
   "models": ["housing_model", "iris_model", "fraud_detection"]
}
```

### `GET /api/models/<model_name>`

Get information about a specific model.

**Example Response:**

```json
{
   "name": "housing_model",
   "config": {
      "task": "regression",
      "n_rounds": 100,
      "learning_rate": 0.1,
      "max_depth": 6,
      "min_child_weight": 1.0
   }
}
```

### `POST /api/predict/<model_name>`

Make predictions using a specific model.

**Request Formats:**

CSV Data:

```json
{
   "csv": "feature1,feature2,feature3\n1.0,2.0,3.0"
}
```

Array Data:

```json
{
   "features": [1.0, 2.0, 3.0]
}
```

**Example Response:**

```json
{
   "prediction": 42.5,
   "model": "housing_model",
   "time_us": 235
}
```

## Running the API Server

### Using Docker:

```bash
# Build and run the API server
docker-compose up boostedpp-api

# Or build and run manually
docker build -t boostedpp-api ./api
docker run -p 8080:8080 -v $(pwd)/models:/models boostedpp-api
```

### Environment Variables:

-  `PORT`: Port to listen on (default: 8080)
-  `MODELS_DIR`: Directory containing model files (default: ./models)

## Example Usage

Using curl:

```bash
# Get version
curl http://localhost:8080/api/version

# List models
curl http://localhost:8080/api/models

# Get model info
curl http://localhost:8080/api/models/housing_model

# Make prediction with CSV data
curl -X POST http://localhost:8080/api/predict/housing_model \
  -H "Content-Type: application/json" \
  -d '{"csv": "feature1,feature2,feature3\n1.0,2.0,3.0"}'

# Make prediction with array data
curl -X POST http://localhost:8080/api/predict/housing_model \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

Using Python:

```python
import requests
import json

# Get version
response = requests.get("http://localhost:8080/api/version")
print(response.json())

# Make prediction
data = {"features": [1.0, 2.0, 3.0]}
response = requests.post(
    "http://localhost:8080/api/predict/housing_model",
    json=data
)
print(response.json())
```

## Building from Source

```bash
mkdir -p build && cd build
cmake ..
make
./boostedpp_api
```
