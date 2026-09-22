# Crop Advisor

## Live Demo

- [Open the Crop Advisor application](https://crop-advisor-frontend.onrender.com)
- [View the API documentation](https://crop-advisor-1.onrender.com/docs)

> **Note:** The application uses Render's free hosting plan. After a period of inactivity, the services may take up to 90 seconds to wake.

Crop Advisor is an end-to-end machine-learning application that recommends crops from soil nutrients and weather measurements. It includes a reproducible cleaning and EDA notebook, three-model comparison, saved scikit-learn pipeline, FastAPI service, and farmer-friendly Streamlit interface.

## What the model uses

| Input | API field | Description |
|---|---|---|
| Nitrogen | `nitrogen` | Soil nitrogen measurement |
| Phosphorus | `phosphorus` | Soil phosphorus measurement |
| Potassium | `potassium` | Soil potassium measurement |
| Soil pH | `ph` | Soil acidity or alkalinity |
| Rainfall | `rainfall` | Rainfall in the source dataset's unit |
| Temperature | `temperature` | Temperature in degrees Celsius |

The source data does not contain humidity or location. The API accepts these fields as optional context, but model version 1 does not use them for prediction.

## Model performance

The supplied train and test CSVs contained the same records in different row orders. To prevent leakage, the project uses one copy, removes exact duplicate observations, and creates new stratified train, validation, and test partitions.

| Model | Validation accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Random Forest | 0.9079 | 0.9524 | **0.9525** |
| HistGradientBoosting | 0.9098 | 0.9485 | 0.9480 |
| Logistic Regression | 0.8809 | 0.9584 | 0.9439 |

Random Forest was selected using validation macro F1. On the untouched test partition, it achieved 0.9079 accuracy, 0.9502 balanced accuracy, and 0.9501 macro F1.

## Architecture

```text
Streamlit frontend
       |
       | POST /predict
       v
FastAPI validation layer
       |
       v
Prediction service
       |
       v
Saved scikit-learn pipeline + metadata
```

The notebook and training modules form a separate offline workflow that produces the model artifacts consumed by the API.

## Repository structure

```text
crop-advisor/
|-- .github/workflows/ci.yml       # Continuous integration
|-- api/                           # FastAPI routes and schemas
|-- app/                           # Streamlit UI and API client
|-- artifacts/                     # Saved model, metadata, and comparison
|-- data/
|   |-- raw/                       # Original CSV files
|   `-- processed/                 # Leakage-safe clean dataset
|-- docs/architecture.md           # Detailed design decisions
|-- notebooks/                     # Cleaning and EDA notebook
|-- reports/figures/               # Generated EDA charts
|-- src/crop_advisor/              # Data, training, and inference modules
|-- tests/                         # Unit and integration tests
|-- Dockerfile
|-- pyproject.toml
`-- requirements.txt
```

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/spencerchux/crop-advisor.git
cd crop-advisor
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Copy the supplied files to:

```text
data/raw/train.csv
data/raw/test.csv
```

The repository already contains the cleaned dataset and trained model artifact. The raw files are needed only when rerunning the cleaning notebook.

## Data cleaning and EDA

Start JupyterLab:

```bash
jupyter lab notebooks/01_data_cleaning_eda.ipynb
```

Run all cells to regenerate:

- `data/processed/crop_recommendation_clean.csv`
- Crop distribution chart
- Feature histograms and box plots
- Feature correlation heatmap
- Crop-level growing-condition profiles

## Train the models

Set the source package path, then run training.

Windows PowerShell:

```powershell
$env:PYTHONPATH="src;."
python -m crop_advisor.train
```

macOS or Linux:

```bash
export PYTHONPATH="src:."
python -m crop_advisor.train
```

Training regenerates:

- `artifacts/model.joblib`
- `artifacts/model_metadata.json`
- `artifacts/model_comparison.csv`

## Run the REST API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

### Prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen": 70,
    "phosphorus": 40,
    "potassium": 45,
    "ph": 5.54,
    "rainfall": 75.32,
    "temperature": 22.676,
    "humidity": 65,
    "location": "Kano, Nigeria"
  }'
```

Example response:

```json
{
  "recommended_crop": "barley",
  "confidence": 1.0,
  "alternatives": [
    {"crop": "wheat", "probability": 0.0},
    {"crop": "coriander", "probability": 0.0}
  ],
  "warnings": [
    "Humidity was accepted for context but is not used by this model.",
    "Location was accepted for context but is not used by this model."
  ],
  "model_name": "random_forest"
}
```

Additional endpoints:

- `GET /health` checks that the API and model are ready.
- `GET /model-info` returns the feature contract and test metrics.

## Run the Streamlit frontend

Keep the API running in one terminal. In another terminal, run:

```bash
streamlit run app/streamlit_app.py
```

The frontend uses `http://localhost:8000` by default. To point it to a deployed API:

Windows PowerShell:

```powershell
$env:CROP_ADVISOR_API_URL="https://your-api.onrender.com"
streamlit run app/streamlit_app.py
```

macOS or Linux:

```bash
export CROP_ADVISOR_API_URL="https://your-api.onrender.com"
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest -q
```

The suite covers dataset validation, stratified splitting, model inference, range warnings, API endpoints, schema validation, and frontend API error handling.

## Docker

Build and run the FastAPI service:

```bash
docker build -t crop-advisor-api .
docker run --rm -p 8000:8000 crop-advisor-api
```

Check the service:

```bash
curl http://localhost:8000/health
```

## Deploy on Render

### API service

1. Push the repository to GitHub.
2. Create a new Render **Web Service**.
3. Select the repository and choose the Docker runtime.
4. Render will build the root `Dockerfile` and supply the `PORT` variable.
5. Confirm `https://YOUR_API_URL/health` returns an `ok` status.

### Streamlit service

Create a second Render Web Service with:

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT`
- Environment variable: `CROP_ADVISOR_API_URL=https://YOUR_API_URL`

## Screenshots

Add repository screenshots after deploying or running the services locally:

1. Save the farmer input form as `docs/screenshots/crop-advisor-form.png`.
2. Save a completed recommendation as `docs/screenshots/crop-advisor-result.png`.
3. Save the FastAPI Swagger page as `docs/screenshots/api-docs.png`.

Then replace this paragraph with Markdown image links so visitors can preview the application directly from GitHub.

## Limitations and responsible use

- Recommendations reflect patterns in the supplied dataset and may not generalize to every farm or region.
- The source does not document the rainfall unit, so users must match the scale used by the dataset.
- Location, humidity, crop prices, planting season, soil texture, pests, and disease pressure are not model features.
- Confidence is a model estimate, not a guarantee of yield or profitability.
- Farmers should confirm important planting decisions with a local agronomist and current field observations.

## Future improvements

- Retrieve current weather from a location-aware weather API.
- Collect humidity, soil texture, season, and regional field outcomes.
- Add calibrated probabilities and uncertainty thresholds.
- Monitor prediction drift and retrain from reviewed observations.
- Translate the farmer interface into local languages.

## License

This project is available under the [MIT License](LICENSE).

