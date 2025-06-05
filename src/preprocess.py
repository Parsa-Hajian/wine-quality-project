# src/preprocess.py

import os
import pandas as pd
import numpy as np

def load_combined_df():
    """
    Load red and white wine CSVs, add a 'color' column, and concatenate into one DataFrame.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    red_path = os.path.join(base_dir, "winequality-red.csv")
    white_path = os.path.join(base_dir, "winequality-white.csv")

    # Read CSVs (they use semicolons as separators)
    red_df = pd.read_csv(red_path, sep=";")
    white_df = pd.read_csv(white_path, sep=";")

    # Add 'color' column
    red_df["color"] = "red"
    white_df["color"] = "white"

    # Concatenate
    combined_df = pd.concat([red_df, white_df], ignore_index=True)
    return combined_df

def create_label(df):
    """
    Add a binary 'good' column: 1 if quality >= 6, else 0.
    Returns a copy of df with 'good'.
    """
    out = df.copy()
    out["good"] = (out["quality"] >= 6).astype(int)
    return out

def stratified_split(df, label_col="good", test_frac=0.2, random_seed=42):
    """
    Perform a stratified train/test split on df based on label_col.
    Returns: (df_train, df_test)
    """
    np.random.seed(random_seed)
    # Shuffle rows
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Split each class separately
    train_indices = []
    test_indices = []
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        n_total = len(subset)
        n_test = int(np.floor(test_frac * n_total))
        # First n_test indices (since subset is already shuffled)
        test_idxs = subset.index[:n_test]
        train_idxs = subset.index[n_test:]
        test_indices.extend(test_idxs)
        train_indices.extend(train_idxs)

    # Create train/test DataFrames
    df_train = df.loc[train_indices].reset_index(drop=True)
    df_test = df.loc[test_indices].reset_index(drop=True)
    return df_train, df_test

def standardize_features(df_train, df_test, feature_cols):
    """
    Standardize (zero-mean, unit-variance) features in df_train and df_test.
    The mean & std are computed on df_train only.
    Returns standardized (X_train, X_test) along with (means, stds).
    """
    # Compute means and stds on train
    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1)  # in case any std is zero
    
    # Standardize
    X_train = (df_train[feature_cols] - means) / stds
    X_test = (df_test[feature_cols] - means) / stds

    return X_train, X_test, means, stds

def save_to_csv(df, path):
    """
    Helper to save DataFrame to a CSV file (without the index).
    """
    df.to_csv(path, index=False)

def main():
    # Step A: Load and label
    combined = load_combined_df()
    labeled = create_label(combined)

    # Step B: Define feature columns (all except 'quality', 'color', 'good')
    exclude_cols = ["quality", "color", "good"]
    all_cols = list(labeled.columns)
    feature_cols = [col for col in all_cols if col not in exclude_cols]

    # Step C: Stratified train/test split
    df_train, df_test = stratified_split(labeled, label_col="good", test_frac=0.2, random_seed=42)

    # Step D: Separate features & labels
    X_train_raw = df_train[feature_cols].copy()
    y_train = df_train["good"].copy()
    X_test_raw = df_test[feature_cols].copy()
    y_test = df_test["good"].copy()

    # Step E: Standardize features
    X_train, X_test, means, stds = standardize_features(df_train, df_test, feature_cols)

    # Step F: Create a "processed" directory inside data/
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Step G: Save processed datasets
    save_to_csv(pd.concat([X_train, y_train.rename("good")], axis=1),
                os.path.join(processed_dir, "train_processed.csv"))
    save_to_csv(pd.concat([X_test, y_test.rename("good")], axis=1),
                os.path.join(processed_dir, "test_processed.csv"))

    # Also save means & stds for later use (as CSVs)
    means.to_csv(os.path.join(processed_dir, "feature_means.csv"), header=True)
    stds.to_csv(os.path.join(processed_dir, "feature_stds.csv"), header=True)

    print("Preprocessing complete.")
    print(f"  - Train set: {X_train.shape[0]} samples")
    print(f"  - Test set : {X_test.shape[0]} samples")
    print(f"  - Features  : {len(feature_cols)} columns")
    print(f"  - Saved to  : {processed_dir}")

if __name__ == "__main__":
    main()
