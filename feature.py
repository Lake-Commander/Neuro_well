# feature.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

def load_data(train_path='train.csv', test_path='test.csv'):
    return pd.read_csv(train_path), pd.read_csv(test_path)

def encode_categoricals(train_df, test_df, columns, save_dir='models'):
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
        joblib.dump(le, f"{save_dir}/{col}_encoder.pkl")
    return train_df, test_df

def fill_missing_values(df):
    # Avoid chained assignment: assign result directly back to the column
    df['Mental Fatigue Score'] = df['Mental Fatigue Score'].fillna(df['Mental Fatigue Score'].mean())
    df['Resource Allocation'] = df['Resource Allocation'].fillna(df['Resource Allocation'].median())
    return df

def calculate_tenure(join_date):
    if pd.isnull(join_date):
        return np.nan
    return (datetime.now() - pd.to_datetime(join_date)).days // 365

def add_tenure_feature(df):
    df['Tenure'] = df['Date of Joining'].apply(calculate_tenure)
    return df.drop(columns=['Date of Joining'])

def normalize_columns(train_df, test_df, columns, save_dir='models'):
    scaler = MinMaxScaler()
    train_df[columns] = scaler.fit_transform(train_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    return train_df, test_df

def save_processed(train_df, test_df, output_dir='processed'):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f"{output_dir}/train_processed.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_processed.csv", index=False)
    print(f"✅ Preprocessing complete. Files saved in '{output_dir}'.")

def preprocess():
    os.makedirs("models", exist_ok=True)
    
    train_df, test_df = load_data()

    # Encode categoricals
    cat_cols = ['Gender', 'Company Type', 'WFH Setup Available']
    train_df, test_df = encode_categoricals(train_df, test_df, cat_cols)

    # Handle missing
    train_df = fill_missing_values(train_df)
    test_df = fill_missing_values(test_df)

    # Add tenure
    train_df = add_tenure_feature(train_df)
    test_df = add_tenure_feature(test_df)

    # Normalize
    num_cols = ['Resource Allocation', 'Designation', 'Tenure']
    train_df, test_df = normalize_columns(train_df, test_df, num_cols)

    # Save results
    save_processed(train_df, test_df)

if __name__ == "__main__":
    preprocess()
