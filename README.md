# 🏠 House Price Prediction Internship Project

## Objective
Predict house prices using property features such as size, number of bedrooms, bathrooms, and location. This project demonstrates skills in data preprocessing, regression modeling, and model evaluation.

## Dataset
**Source:** House Price Prediction Dataset (available on Kaggle)  
**Size after cleaning:** 365 rows × 13 columns

## Features

### Numerical Features
- area – House size in square feet  
- bedrooms – Number of bedrooms  
- bathrooms – Number of bathrooms  
- stories – Number of floors  
- parking – Number of parking spaces  

### Categorical Features
- mainroad – House near main road (yes/no)  
- guestroom – Availability of guestroom (yes/no)  
- basement – Presence of basement (yes/no)  
- hotwaterheating – Presence of hot water heating (yes/no)  
- airconditioning – Presence of air conditioning (yes/no)  
- prefarea – Preferred area (yes/no)  
- furnishingstatus – Furnishing status (furnished/semi-furnished/unfurnished)  

### Target Variable
- price – Price of the house

## Methodology

### 1️⃣ Data Preprocessing
- Outlier Removal: Applied IQR method on numerical columns to remove extreme values.  
- Feature Scaling: StandardScaler used for numerical features.  
- Categorical Encoding: OneHotEncoder applied for categorical features.  
- Train-Test Split: 80% training, 20% testing.

### 2️⃣ Model Training
- Linear Regression model used inside a Pipeline combining preprocessing and modeling.  
- Model trained on training data and predictions made on test data.

### 3️⃣ Model Evaluation
**Metrics:**  
- MAE: 704,466  
- RMSE: 1,036,438  
- R² Score: 0.497  

**Interpretation:**  
- MAE & RMSE measure the average error of predictions.  
- R² Score ~0.5 indicates the model explains ~50% of the variance in house prices.

### 4️⃣ Visualization
Actual vs Predicted Prices:

```python
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()
