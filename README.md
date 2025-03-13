# car_price_prediction
A Flask-based web app for car price prediction using Ridge Regression
# Car Price Prediction Using Ridge Regression

## Overview
This project implements a **machine learning model** to predict the price of a used car based on various input features like mileage, year, engine size, transmission type, and fuel type. The model uses **Ridge Regression**, along with advanced feature engineering techniques, to achieve high accuracy and robustness in price estimation.

The web application is built using **Flask**, providing an interactive UI where users can input car details and get a predicted price instantly.

---

## Dataset
The dataset consists of used car listings with the following key features:
- **Make & Model**: Brand and model of the car.
- **Year**: Year of manufacture.
- **Mileage**: Number of miles driven.
- **Tax**: Road tax for the vehicle.
- **MPG (Miles Per Gallon)**: Fuel efficiency.
- **Engine Size**: The engine displacement in liters.
- **Transmission**: Type of gear system (Manual, Automatic, Semi-Auto).
- **Fuel Type**: Petrol, Diesel, Hybrid, Electric.
- **Price**: The target variable (log-transformed for better predictions).

---

## Model Development

### 1. **Data Preprocessing & Feature Engineering**
- **Outlier Removal**: Applied **Interquartile Range (IQR)** method to remove extreme values from numerical features.
- **Log Transformation**: 
  - The target variable (**price**) is transformed using **log1p** to stabilize variance and improve regression performance.
  - The **mileage** feature is also log-transformed to correct skewness.
- **Feature Scaling**: Standardized numerical features using **StandardScaler**.
- **One-Hot Encoding**: Categorical features (**Make, Model, Transmission, Fuel Type**) are transformed into numerical representations using **OneHotEncoder**.
- **Feature Selection**: Applied **SelectKBest** (using Mutual Information Regression) to keep the most relevant features.
- **Polynomial Feature Expansion**: Applied **PolynomialFeatures** (degree=2, interaction-only) to capture feature interactions and improve model accuracy.

### 2. **Model Training (Ridge Regression)**
- Chose **Ridge Regression** due to its robustness against multicollinearity and overfitting.
- **Hyperparameter tuning**: Fine-tuned the regularization parameter **alpha=0.5** to balance bias and variance.
- **Train-Test Split**: Data split into **80% training** and **20% testing** sets.

### 3. **Performance Evaluation**
The model's performance is evaluated using multiple metrics:
- **Mean Absolute Error (MAE):** Measures average absolute prediction error.
- **Mean Squared Error (MSE) & Root Mean Squared Error (RMSE):** Measures variance and overall prediction accuracy.
- **R-squared Score (R²):** Indicates how well the model explains the variance in car prices.
- **Mean Absolute Percentage Error (MAPE):** Measures percentage error to compare across different price ranges.
- **Cross-validation**: 5-fold **cross-validation** ensures stability and generalization.

| Metric  | Value  |
|---------|--------|
| MAE     | ~$800  |
| MSE     | ~1.2M  |
| RMSE    | ~$1,100 |
| R² Score| ~0.92  |
| MAPE    | ~11%   |

---

## Web Application (Flask UI)
The **Flask** web app allows users to:
1. **Select Car Make & Model** dynamically.
2. **Input Car Specifications** (Year, Mileage, Engine Size, etc.).
3. **Get Predicted Price** instantly after submission.

### API Endpoints:
- `/` - Home page with form input.
- `/get_models/<make>` - Returns available car models based on selected make.
- `/predict` - Accepts input data, processes it, and returns a predicted price.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/SRIVIJAYKRISHNA/car_price_prediction.git
   cd car_price_prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000/` in your browser.

---

## Future Enhancements
- **Deploy on Cloud** (AWS, GCP, or Heroku).
- **Improve Model Performance** (Ensemble Models, Feature Engineering).
- **Add More Features** (Car Condition, Owner History, Market Trends).

---

## Conclusion
This project successfully predicts car prices using **Ridge Regression** with advanced preprocessing techniques. The Flask-based UI makes it accessible for users, and the model achieves a high level of accuracy with optimized feature selection and transformation.
