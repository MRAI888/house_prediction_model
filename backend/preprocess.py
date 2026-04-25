import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from utils import log_message


def _normalise_columns(columns):
    """
    Convert columns to a consistent snake_case style.
    Example:
    'AveragePrice' -> 'averageprice'
    '1m%Change' -> '1mpctchange'
    'Region Name' -> 'region_name'
    """
    cleaned = []
    for col in columns:
        col = str(col).strip().lower()
        col = col.replace('%', 'pct')
        col = re.sub(r'[^a-z0-9]+', '_', col)
        col = re.sub(r'_+', '_', col).strip('_')
        cleaned.append(col)
    return cleaned


def load_and_clean(filepath):
    """
    Load UK_House_Prices.csv and prepare it for modelling.

    Supports datasets with columns like:
    - Date
    - AveragePrice
    - RegionName
    - AreaCode
    - SalesVolume
    - DetachedPrice / FlatPrice / etc.
    """
    df = pd.read_csv(filepath)
    log_message(f"Loaded raw data: {df.shape}")

    # Standardise column names
    df.columns = _normalise_columns(df.columns)

    # ----------------------------
    # Auto-detect key columns
    # ----------------------------

    # Date column
    date_candidates = [
        'date', 'sale_date', 'date_of_transfer',
        'transferdate', 'transaction_date'
    ]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise KeyError(f"No recognised date column. Columns: {df.columns.tolist()}")

    # Price / target column
    price_candidates = [
        'price', 'sale_price', 'sold_price', 'value',
        'averageprice', 'average_price', 'avgprice', 'avg_price'
    ]
    price_col = next((c for c in price_candidates if c in df.columns), None)
    if price_col is None:
        raise KeyError(f"No recognised price column. Columns: {df.columns.tolist()}")

    df.rename(columns={date_col: 'Date', price_col: 'Price'}, inplace=True)

    # Location columns
    location_map = {
        'town': 'Town',
        'city': 'Town',
        'region': 'Town',
        'regionname': 'Town',
        'county': 'County',
        'district': 'District',
        'postcode_area': 'District',
        'areacode': 'AreaCode',
        'area_code': 'AreaCode'
    }
    for old, new in location_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Property type
    type_candidates = ['property_type', 'propertytype', 'type']
    for col in type_candidates:
        if col in df.columns:
            df.rename(columns={col: 'property_type'}, inplace=True)
            break

    # New build
    new_build_candidates = ['new_build', 'newbuild', 'new_build_flag']
    for col in new_build_candidates:
        if col in df.columns:
            df.rename(columns={col: 'new_build'}, inplace=True)
            break

    # Tenure / freehold
    tenure_candidates = ['freehold', 'tenure', 'freehold_leasehold']
    for col in tenure_candidates:
        if col in df.columns:
            df.rename(columns={col: 'freehold_raw'}, inplace=True)
            break

    # ----------------------------
    # Basic cleaning
    # ----------------------------
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    df = df.dropna(subset=['Date', 'Price']).copy()
    df = df[df['Price'] > 1000].copy()

    # Date features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter

    # ----------------------------
    # Encode categoricals
    # ----------------------------

    # New build
    if 'new_build' in df.columns:
        df['new_build'] = (
            df['new_build']
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                'y': 1, 'n': 0,
                'yes': 1, 'no': 0,
                '1': 1, '0': 0,
                'true': 1, 'false': 0
            })
            .fillna(0)
            .astype(int)
        )
    else:
        df['new_build'] = 0

    # Freehold
    if 'freehold_raw' in df.columns:
        df['freehold'] = df['freehold_raw'].apply(
            lambda x: 1 if str(x).strip().lower() in ['freehold', 'f', '1', 'yes', 'true'] else 0
        )
    else:
        df['freehold'] = 0

    # Property type one-hot encoding
    if 'property_type' in df.columns:
        mapping = {
            'd': 'Detached',
            'detached': 'Detached',
            's': 'Semi',
            'semi': 'Semi',
            'semi-detached': 'Semi',
            'semi_detached': 'Semi',
            't': 'Terraced',
            'terraced': 'Terraced',
            'f': 'Flat',
            'flat': 'Flat',
            'apartment': 'Flat'
        }

        df['property_type'] = (
            df['property_type']
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .fillna('Other')
        )

        df = pd.get_dummies(df, columns=['property_type'], prefix='type')
    else:
        df['type_Detached'] = 0
        df['type_Semi'] = 0
        df['type_Terraced'] = 0
        df['type_Flat'] = 0
        df['type_Other'] = 0

    # Compatibility columns
    df['type_D'] = df.get('type_Detached', 0)
    df['type_S'] = df.get('type_Semi', 0)
    df['type_T'] = df.get('type_Terraced', 0)
    df['type_F'] = df.get('type_Flat', 0)

    # ----------------------------
    # Target encoding for location columns
    # ----------------------------
    for col in ['Town', 'County', 'District', 'AreaCode']:
        if col in df.columns:
            means = df.groupby(col)['Price'].mean()
            df[f'{col}_enc'] = df[col].map(means).fillna(df['Price'].mean())
        else:
            df[f'{col}_enc'] = df['Price'].mean()

    # ----------------------------
    # Drop unused text columns
    # ----------------------------
    drop_text = [
        'Town', 'County', 'District', 'AreaCode',
        'postcode', 'street', 'locality',
        'freehold_raw',
        'type_Detached', 'type_Semi', 'type_Terraced', 'type_Flat', 'type_Other'
    ]

    existing_drop = [col for col in drop_text if col in df.columns]
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)

    # ----------------------------
    # Fill missing numeric values
    # ----------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    log_message(f"Preprocessed data shape: {df.shape}")
    return df


def prepare_features(df, target='Price'):
    """Separate features and target."""
    exclude = [target, 'Date']
    drop_cols = [c for c in exclude if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    y = df[target].copy()
    return X, y


def scale_features(X_train, X_test):
    """Scale numeric columns only."""
    scaler = RobustScaler()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    X_train_num = X_train[numeric_cols].copy()
    X_test_num = X_test[numeric_cols].copy()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_num),
        columns=numeric_cols,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_num),
        columns=numeric_cols,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler