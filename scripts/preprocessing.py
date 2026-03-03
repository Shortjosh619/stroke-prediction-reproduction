"""
Data Preprocessing for Stroke Prediction Dataset

This script performs preprocessing steps including:
- Missing value imputation (mean for BMI)
- Categorical encoding (label and one-hot)
- Feature scaling (Min-Max normalization [0,1])
- Train-test split (stratified 80/20)
- Random oversampling (training data only)

Author: Joshua Ajemiri
Date: February 2026
"""

import pandas as pd
import numpy as np 
from pathlib import Path 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

DATA_PATH = Path(r"C:\stroke-prediction-reproduction\data\raw\healthcare-dataset-stroke-data.csv")

RANDOM_STATE = 9
TEST_SIZE = 0.20



def load_data(filepath):

    df = pd.read_csv(filepath)
    
    bmi_mean = np.nanmean(df["bmi"])
    print(f"BMI mean (excluding missing values): {bmi_mean:.2f}")

    df["bmi"] = df["bmi"].fillna(bmi_mean)
    missing = df["bmi"].isnull().sum()

    print(f"Missing BMI values now: {missing}")
    print("Imputation done.\n")

    return df


def encode_categoricals(
        df,
        id_col=None,
        binary_cols=None,
        multi_cols=None,
        ordinal_cols=None,
        drop_first=True,
        ):
    
    df_encoded = df.copy()
    if id_col and id_col in df_encoded.columns:
        df_encoded.drop(id_col, axis=1, inplace=True)
        print(f"dropped {id_col} column")

    #Label encoding for binary
    if binary_cols:
        print("\n Label Encoding for binary columns")
        for col in binary_cols:
            if col not in df_encoded.columns:
                raise KeyError(f"Column {col} not in dataframe")
            
            unique_vals = df_encoded[col].dropna().unique()
            if len(unique_vals) != 2:
                print(f"Warning: '{col}' has {len(unique_vals)} unique values. Expected 2.")
            
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            df_encoded[col] = df_encoded[col].map(mapping)
            print(f"{col}: mapped to {unique_vals}")

    #Label encodin for ordinal
    if ordinal_cols: 
        print("\n Label Encoding for ordinal columns")
        for col, categories in ordinal_cols.items():
            if col not in df_encoded.columns:
                raise KeyError(f"Column {col} not found in dataframe")
        
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            df_encoded[col] = df_encoded[col].map(mapping)
            print(f"{col}: mapped to {categories}")

    #One-Hot encoding for multi-class 
    if multi_cols:
        print("\n One Hot Encoding for multi-class columns")
        for col in multi_cols:
            if col not in df_encoded.columns:
                raise KeyError(f"Warning: {col} not found in dataframe")
            dummy = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
            df_encoded = pd.concat([df_encoded, dummy], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)

    print(f"Original features: {df.shape[1]}")
    print(f"Encoded features: {df_encoded.shape[1]}")

    return df_encoded



def scale_features(X_train, X_test, return_df=True):

    scaler = MinMaxScaler()
   
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    if return_df:

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    
    return X_train_scaled, X_test_scaled, scaler

def split_data(df):

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state= RANDOM_STATE, stratify=y)

    print(f"Random State: {RANDOM_STATE}")
    print(f"Split Ratio: {(1-TEST_SIZE)*100:.2f}% train, {TEST_SIZE*100:.2f}% test")
    print(f"\nTraining set observations (N= {X_train.shape[0]})")
    print(f"No stroke: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train) * 100:.2f}%)")
    print(f"Stroke: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train) * 100:.2f}%)")

    print(f"Test set observations (N= {y_test.shape[0]})")
    print(f"No stroke: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test) * 100:.2f}%)")
    print(f"Stroke: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test) * 100:.2f}%)")

    return X_train, X_test, y_train, y_test


def apply_oversampling(X_train, y_train):

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    #Added some print statemtns so i can check console output to confirm oversampling worked
    print("Before oversampling:")
    print(f"No stroke: {(y_train == 0).sum()}")
    print(f"Stroke: {(y_train == 1).sum()}")
    print(f"Ratio: {(y_train == 0).sum() / (y_train == 1).sum()}")

    print("After oversampling:")
    print(f"No stroke: {(y_train_resampled == 0).sum()}")
    print(f"Stroke: {(y_train_resampled == 1).sum()}")
    print(f"Ratio: {(y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()}")

    return X_train_resampled, y_train_resampled

def save_preprocessed_data(X_train, X_test, y_train, y_test):

    data_dir = Path(r"C:\stroke-prediction-reproduction\data\processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(data_dir / "X_train.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_train.to_csv(data_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(data_dir/ "y_test.csv", index=False, header=True)

    print(f"saved to {data_dir}")



def main():
    df = load_data(DATA_PATH)
    
    df_encoded = encode_categoricals(df, id_col="id", binary_cols=["ever_married", "Residence_type"],
                                      multi_cols= ["work_type", "smoking_status"], ordinal_cols={"gender": ["Male", "Female", "Other"]},
                                    drop_first=True)
    
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    X_train_resampled, y_train_resampled = apply_oversampling(X_train_scaled, y_train)

    save_preprocessed_data(X_train_resampled, X_test_scaled, y_train_resampled, y_test)

    print("Preprocessing complete")

if __name__ == "__main__":
    main()