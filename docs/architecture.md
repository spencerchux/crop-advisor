# Crop Advisor architecture

## Scope

Crop Advisor recommends a crop from soil measurements and weather inputs. The supplied dataset contains `N`, `P`, `K`, `pH`, `rainfall`, `temperature`, and the target `Crop`.

The supplied data does not contain `humidity` or `location`. The first model version will therefore use the six features that are actually present. The API and frontend will keep location optional so a later weather-service integration can convert location into model-supported weather values without inventing training data.

## System flow

1. A farmer enters soil and weather measurements in the Streamlit interface.
2. Streamlit sends a JSON request to the FastAPI prediction endpoint.
3. FastAPI validates ranges and passes the ordered features to the saved scikit-learn pipeline.
4. The pipeline applies the same preprocessing used during training and predicts a crop.
5. The API returns the recommended crop, confidence when supported, and model metadata.

## Repository structure

```text
crop-advisor/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- api/
|   |-- __init__.py
|   |-- main.py
|   |-- schemas.py
|   `-- services.py
|-- app/
|   `-- streamlit_app.py
|-- artifacts/
|   |-- model.joblib
|   `-- model_metadata.json
|-- data/
|   |-- raw/
|   |   |-- train.csv
|   |   `-- test.csv
|   `-- processed/
|-- docs/
|   `-- architecture.md
|-- notebooks/
|   `-- 01_data_cleaning_eda.ipynb
|-- src/
|   `-- crop_advisor/
|       |-- __init__.py
|       |-- config.py
|       |-- data.py
|       |-- features.py
|       |-- train.py
|       `-- inference.py
|-- tests/
|   |-- test_api.py
|   |-- test_data.py
|   `-- test_inference.py
|-- .dockerignore
|-- .gitignore
|-- Dockerfile
|-- LICENSE
|-- README.md
|-- pyproject.toml
`-- requirements.txt
```

## Component responsibilities

- `notebooks/`: reproducible data cleaning and exploratory analysis.
- `src/crop_advisor/`: reusable data, training, and inference code. Business logic stays out of the notebook and API routes.
- `artifacts/`: generated model pipeline and machine-readable metadata. These are produced by training rather than edited manually.
- `api/`: FastAPI transport layer, request validation, health check, and prediction endpoint.
- `app/`: farmer-facing Streamlit client that calls the API.
- `tests/`: unit and API tests run locally and in GitHub Actions.
- `data/raw/`: immutable copies of the supplied CSV files. Generated clean data belongs in `data/processed/`.

## Modeling plan

We will compare at least Logistic Regression, Random Forest, and HistGradientBoosting using the same stratified validation split and metrics. The final saved artifact will be a single scikit-learn `Pipeline`, which prevents training-serving preprocessing drift.

The CSV column `Unnamed: 0` is an exported row index and will be removed. Before modeling, we will also verify whether the provided train and test files overlap. If they are duplicates or substantially overlap, we will deduplicate the source and create a new stratified train/validation/test split to prevent leakage.

## API contract

Planned endpoint: `POST /predict`

```json
{
  "nitrogen": 70,
  "phosphorus": 40,
  "potassium": 45,
  "ph": 5.54,
  "rainfall": 75.32,
  "temperature": 22.676,
  "humidity": null,
  "location": null
}
```

`humidity` and `location` are reserved optional fields and will not affect predictions until supported by trained data or an explicit external-data feature pipeline.

## Deployment shape

The default deployment target will be one Docker image that runs FastAPI. Streamlit can run as a second Render service from the same repository. This keeps the API independently testable and allows the frontend to be replaced later without retraining the model.
