
import pandas as pd
from pathlib import Path

file_path = Path("data/creditcard.csv")

if not file_path.exists():
    print("Dataset not found!")
    print("Please put creditcard.csv inside the data folder.")
else:
    df = pd.read_csv(file_path)

    print("\n--- FIRST 5 TRANSACTIONS ---")
    print(df.head())

    print("\n--- DATASET SIZE ---")
    print("Rows and columns:", df.shape)

    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum().sum())

    print("\n--- FRAUD CLASS COUNTS ---")
    print(df["Class"].value_counts())

    print("\n--- FRAUD PERCENTAGE ---")
    print(round(df["Class"].mean() * 100, 4), "%")
