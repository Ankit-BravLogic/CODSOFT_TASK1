
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_dataset(filename, columns):
    path = DATA_DIR / filename
    return pd.read_csv(
        path,
        sep=r"\s*:::\s*",
        engine="python",
        header=None,
        names=columns,
        on_bad_lines="error",
    )

train_columns = ["id", "title", "genre", "description"]
test_columns = ["id", "title", "description"]

train_df = load_dataset("train_data.txt", train_columns)
test_df = load_dataset("test_data.txt", test_columns)
solution_df = load_dataset("test_data_solution.txt", train_columns)

print("\n--- DATASET SHAPES ---")
print("Training:", train_df.shape)
print("Test:", test_df.shape)
print("Test solution:", solution_df.shape)

print("\n--- TRAINING SAMPLE ---")
print(train_df.head(3).to_string(index=False))

print("\n--- MISSING VALUES ---")
print(train_df.isnull().sum())

print("\n--- GENRE DISTRIBUTION ---")
print(train_df["genre"].value_counts())

print("\n--- DUPLICATE CHECK ---")
print("Duplicate rows:", train_df.duplicated().sum())
print("Duplicate IDs:", train_df["id"].duplicated().sum())
