

# Rental Trend Predictor

This project generates a synthetic dataset of monthly rental prices for five major US cities and uses linear regression to predict the next month's rent. It also visualizes rent trends using Seaborn and Matplotlib.

## Features

- Simulates rental prices with trends and seasonality
- Uses Linear Regression for forecasting
- Evaluates model with MSE and R² Score
- Visualizes rental trends per city

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn (LinearRegression)
- Seaborn & Matplotlib

## How to Run

1. Clone the repo:

git clone https://github.com/yourusername/rental-trend-predictor.git

2. Install dependencies:

pip install -r requirements.txt

3. Open `rental_predictor.py`

4. Run using:

python rental_predictor.py

## Results

- Predicts rent for next month in each city
- Visual trend lines of rent over 2 years

## Deployment

This project runs locally via Python. No deployment necessary.

## Author

Shailusha T

---

## How It Works

### 1. Data Generation
- Created synthetic rental data for 24 months using:
- A random base rent per city
- A linear trend to simulate growth
- Seasonal variations using sine waves
- Random noise for realism

### 2. Preprocessing
- Converted month into a numerical index (`MonthIndex`) for regression.
- Data split by city to model each city individually.

### 3. Modeling
- Used `LinearRegression` from `scikit-learn`.
- Scaled input features using `StandardScaler`.
- Trained models for each city to predict rent trends.

### 4. Prediction
- Predicted the rent for the next (25th) month using trained models.

### 5. Evaluation
- Measured performance using:
- Mean Squared Error (MSE)
- R² Score

### 6. Visualization
- Created line plots for rent trends over time using Seaborn.
- Each city's rent trend displayed with date labels.


## Author



Shailusha T
