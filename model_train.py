import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("C:/Users/marreddyGowthamreddy/Desktop/consumption_app/Power consumption.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Features
X = df[['Temperature', 'Humidity', 'WindSpeed']]

# Targets
y1 = df['PowerConsumption_Zone1']
y2 = df['PowerConsumption_Zone2']
y3 = df['PowerConsumption_Zone3']

# Train-test split
X_train, _, y1_train, _ = train_test_split(X, y1, test_size=0.2, random_state=42)
_, _, y2_train, _ = train_test_split(X, y2, test_size=0.2, random_state=42)
_, _, y3_train, _ = train_test_split(X, y3, test_size=0.2, random_state=42)

# Train models
model1 = RandomForestRegressor().fit(X_train, y1_train)
model2 = RandomForestRegressor().fit(X_train, y2_train)
model3 = RandomForestRegressor().fit(X_train, y3_train)

# Save models
joblib.dump(model1, "rf_zone1.pkl")
joblib.dump(model2, "rf_zone2.pkl")
joblib.dump(model3, "rf_zone3.pkl")
