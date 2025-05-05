
pip install pandas numpy matplotlib seaborn scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(42)
months = pd.date_range(start='2022-01-01', periods=24, freq='M')
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
data = []
for city in cities:
    base_rent = np.random.randint(1000, 3000)
    trend = np.random.uniform(10, 50)
    seasonal = 100 * np.sin(np.linspace(0, 4 * np.pi, len(months)))
    noise = np.random.normal(0, 50, len(months))
    rents = base_rent + trend * np.arange(len(months)) + seasonal + noise
    for i in range(len(months)):
        data.append([months[i], city, rents[i]])
df = pd.DataFrame(data, columns=['Month', 'City', 'Rent'])
df['MonthIndex'] = df['Month'].dt.month + 12 * (df['Month'].dt.year - 2022)
predictions = {}
for city in cities:
    city_df = df[df['City'] == city]
    X = city_df[['MonthIndex']]
    y = city_df['Rent']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    next_month_index = city_df['MonthIndex'].max() + 1
    next_month_scaled = scaler.transform([[next_month_index]])
    next_rent = model.predict(next_month_scaled)[0]
    predictions[city] = round(next_rent, 2)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\n{city}:\nMSE: {mse:.2f}, R²: {r2:.2f}, Predicted Rent for Next Month: ${next_rent:.2f}")
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Month', y='Rent', hue='City')
plt.title('Monthly Rent Trends by City')
plt.xlabel('Month')
plt.ylabel('Rent ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
