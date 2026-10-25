
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "genre_classifier.pkl"

model = joblib.load(MODEL_PATH)

print("Movie Genre Classification")
print("Enter a movie plot below.")
print("Type 'exit' to quit.\n")

while True:
    description = input("Movie plot: ").strip()

    if description.lower() == "exit":
        print("Goodbye!")
        break

    if not description:
        print("Please enter a movie description.\n")
        continue

    predicted_genre = model.predict([description])[0]
    print(f"Predicted genre: {predicted_genre}\n")
