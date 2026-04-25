# UK House Price Prediction System

End-to-end machine learning system for predicting UK property prices with comparative modelling, explainability, and drift monitoring.

## Quick Start

1. Install dependencies: `pip install -r backend/requirements.txt`
2. Place Kaggle datasets in `data/raw/`
3. Train models: `python backend/train.py`
4. Run API: `python backend/app.py`
5. Open `frontend/index.html` in browser (or via Flask at `http://localhost:5000`)

## Features

- Five regression models compared (XGBoost selected)
- Web dashboard for predictions
- PSI‑based drift detection
- Confusion matrix for binned prices
- PlantUML diagrams for documentation

## Datasets

- UK House Price Prediction 2015‑2024 (Kaggle)
- Nationwide Monthly UK House Prices (Kaggle)