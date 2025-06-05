# src/load_and_inspect.py

import os
import pandas as pd

def main():
    # 1. Define file paths
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    red_path = os.path.join(data_dir, "winequality-red.csv")
    white_path = os.path.join(data_dir, "winequality-white.csv")

    # 2. Load each CSV into a DataFrame
    print("Loading red wine data...")
    red_df = pd.read_csv(red_path, sep=";")
    print("Loading white wine data...")
    white_df = pd.read_csv(white_path, sep=";")

    # 3. Add a 'color' column to distinguish them
    red_df["color"] = "red"
    white_df["color"] = "white"

    # 4. Combine into a single DataFrame
    combined_df = pd.concat([red_df, white_df], ignore_index=True)

    # 5. Display basic information
    print("\n=== Combined DataFrame Info ===")
    print(combined_df.info())

    print("\n=== Combined DataFrame Description (numerical) ===")
    print(combined_df.describe())

    # 6. Check class balance (good vs. bad)
    # Create a binary column: 1 if quality >= 6 (good), else 0 (bad)
    combined_df["good"] = (combined_df["quality"] >= 6).astype(int)
    print("\n=== Good vs. Bad Counts ===")
    print(combined_df["good"].value_counts())

    # 7. Optional: Save a small preview CSV (first 5 rows)
    preview_path = os.path.join(os.path.dirname(__file__), "..", "data", "preview.csv")
    combined_df.head().to_csv(preview_path, index=False)
    print(f"\nSaved first 5 rows to: {preview_path}")

if __name__ == "__main__":
    main()
