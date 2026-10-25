# 🎬 Movie Genre Classification

A Machine Learning and Natural Language Processing (NLP) project developed as part of the **CodSoft Machine Learning Internship — Project 3**.

## 📌 Overview

This project predicts a movie's likely genre from its title and plot description using text processing and supervised machine learning.

The application uses TF-IDF to convert text into numerical features and Logistic Regression to classify descriptions into one of 27 movie genres.

## ✨ Features

- Predict a movie genre from a plot description.
- Text feature extraction using TF-IDF.
- Classification across 27 genres.
- Interactive web interface built with Streamlit.
- Dataset inspection, model evaluation, and visual reports.

## 🛠️ Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib and Seaborn
- Streamlit
- Joblib

## 📊 Dataset

The dataset contains 54,214 labeled training records across 27 genres, along with separate test data and test labels.

The original dataset files are excluded from this repository because of their size. Place the dataset files in the `data/` directory before running the training scripts.

Expected files:

- `train_data.txt`
- `test_data.txt`
- `test_data_solution.txt`
- `description.txt`

The files use `:::` as the field separator.

## 📈 Model Performance

The initial Logistic Regression model achieved **58.57% validation accuracy** on a stratified 80/20 training-validation split.

- Validation accuracy: 58.57%
- Macro F1-score: 0.4204
- Weighted F1-score: 0.5879

These are validation results, not results on the separate test dataset.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CODSOFT_TASK1
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place the four dataset files in the `data/` directory.

### 5. Train the model

```bash
python src/train_model.py
```

### 6. Launch the web application

```bash
streamlit run app.py
```

The application uses the saved model at `models/genre_classifier.pkl`. Ensure that you have trained the model before starting the app.

## 📁 Project Structure

```text
CODSOFT_TASK1/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
├── notebooks/
├── reports/
└── src/
    ├── inspect_data.py
    ├── train_model.py
    ├── compare_models.py
    └── predict_genre.py
```

## 🎯 Learning Outcomes

- Text preprocessing and feature extraction.
- Multiclass text classification.
- Model evaluation using accuracy and F1-scores.
- Building an interactive machine learning application.
- Organizing a reproducible Python project.

## 👨‍💻 Internship

**CodSoft Machine Learning Internship — Project 3: Movie Genre Classification**

Developed for educational purposes.
