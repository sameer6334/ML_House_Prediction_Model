# 🏡 California Housing Price Prediction — End-to-End ML Pipeline

This project focuses on building a **complete machine learning pipeline** to predict **California housing prices** using **Scikit-Learn** and regression algorithms.

---

## 🚀 Project Overview

In this project, I:

- Loaded and preprocessed the dataset (`housing.csv`) with:
  - Handling missing values using `SimpleImputer`
  - Feature scaling
  - Encoding categorical features using a **custom pipeline**
- Applied **Stratified Split** to maintain distribution of income categories between train and test sets.
- Trained and evaluated multiple machine learning models:
  - **Linear Regression**
  - **Decision Tree Regressor**
  - **Random Forest Regressor**
- Used **Cross-Validation** to compare model performance using **RMSE (Root Mean Squared Error)**.
- Discovered that **Random Forest Regressor** gave the **best and most stable performance**.

---

## 🧠 Model Deployment Logic

I developed a script that:

- Trains the **Random Forest model** and **saves it using `joblib`**.
- Uses an **if-else logic** to **skip retraining if the model is already saved**.
- Reads new input data from `input.csv`, applies the pipeline, and **predicts `median_house_value`**.
- Exports predictions to **`output.csv`** for easy inference.

---

## ✅ Key Features

- Complete ML pipeline using **ColumnTransformer + Pipeline**
- **Automated preprocessing** for both training and inference stages
- **Joblib-based model persistence**
- **Production-style inference workflow**
- Simply drop your `input.csv` and run → **Get predictions instantly**

---

# ⚙️ Tech Stack

| Component           | Tool/Library      |
|--------------------|------------------|
| Language           | Python            |
| ML Framework       | Scikit-Learn      |
| Model Saving       | Joblib            |
| Dataset            | California Housing Dataset |
| Evaluation Metric  | RMSE (Root Mean Squared Error) |

---

## 🤝 Contributions & Suggestions

If you have ideas for improvements or want to collaborate, feel free to open an issue or fork the repository!

---

### ⭐ If this project helped you, consider dropping a star on the repository!

## 📂 Project Structure

│-- data/
│ │-- housing.csv
│ │-- input.csv
│-- notebooks/
│ │-- exploration.ipynb
│-- models/
│ │-- model.pkl (auto-generated)
│-- output/
│ │-- output.csv (auto-generated)
│-- src/
│ │-- train_and_predict.py
│-- README.md

