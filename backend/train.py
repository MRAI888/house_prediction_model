import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split

from preprocess import load_and_clean, prepare_features, scale_features
from models import get_models, evaluate_model
from utils import save_model, log_message


def main():
    # Make paths reliable no matter where you run the script from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data = os.path.join(base_dir, '..', 'data', 'raw', 'UK_House_Prices.csv')
    model_dir = os.path.join(base_dir, '..', 'models')
    results_dir = os.path.join(base_dir, '..', 'results')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(raw_data):
        log_message(f"ERROR: Dataset not found at {raw_data}")
        log_message("Please place UK_House_Prices.csv in data/raw/")
        sys.exit(1)

    # 1. Load and preprocess
    log_message("Loading and cleaning data...")
    df = load_and_clean(raw_data)
    log_message(f"Data shape after cleaning: {df.shape}")

    # Quick validation: target column must exist
    if 'Price' not in df.columns:
        log_message("ERROR: 'Price' column missing after preprocessing.")
        sys.exit(1)

    X, y = prepare_features(df)

    # 2. Split data
    if 'year' in df.columns and df['year'].max() > 2022:
        # Temporal split for newer data
        mask_train = df['year'] <= 2022
        X_train, X_test = X[mask_train], X[~mask_train]
        y_train, y_test = y[mask_train], y[~mask_train]
        log_message("Using temporal split (train ≤2022, test >2022).")
    else:
        # Fallback random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        log_message("Using random 80/20 split.")

    log_message(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 4. Train models
    models = get_models()
    results = []

    for name, model in models.items():
        log_message(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        metrics['Model'] = name
        results.append(metrics)

        print(
            f"{name}: "
            f"MAE=£{metrics['MAE']:,.0f}, "
            f"RMSE=£{metrics['RMSE']:,.0f}, "
            f"R²={metrics['R2']:.3f}"
        )

    # 5. Save best model (XGBoost) and artifacts
    best_model = models['XGBoost']
    save_model(best_model, os.path.join(model_dir, 'xgboost_model.pkl'))
    save_model(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(X_train_scaled.columns.tolist(), os.path.join(model_dir, 'features.pkl'))
    log_message("Model, scaler, and feature list saved.")

    # 6. Save comparison results
    pd.DataFrame(results).to_csv(
        os.path.join(results_dir, 'model_comparison.csv'),
        index=False
    )
    log_message("Training complete!")


if __name__ == '__main__':
    main()