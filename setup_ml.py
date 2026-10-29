import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Generating advanced synthetic data (with Time-Series & Regions) for Indian Banks...")

branches = [
    {"name": "SBI - Mumbai", "lat": 18.9256, "lon": 72.8242, "region": "West"},
    {"name": "HDFC - Delhi", "lat": 28.6315, "lon": 77.2167, "region": "North"},
    {"name": "ICICI - Bangalore", "lat": 12.9716, "lon": 77.5946, "region": "South"},
    {"name": "Axis - Kolkata", "lat": 22.5535, "lon": 88.3498, "region": "East"},
    {"name": "PNB - Chennai", "lat": 13.0405, "lon": 80.2337, "region": "South"},
    {"name": "Kotak - Pune", "lat": 18.5362, "lon": 73.8939, "region": "West"},
    {"name": "BoB - Ahmedabad", "lat": 23.0225, "lon": 72.5714, "region": "West"},
    {"name": "IndusInd - Hyderabad", "lat": 17.3850, "lon": 78.4867, "region": "South"}
]

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
data = []
np.random.seed(42)

for b in branches:
    # Adding a base efficiency that slightly grows over months to simulate realistic trends
    base_efficiency = np.random.uniform(60, 80)
    for month_idx, month in enumerate(months):
        staff = np.random.randint(15, 50)
        transactions = np.random.randint(1500, 7000)
        satisfaction = round(np.random.uniform(3.5, 4.9), 1)
        
        # Monthly growth trend
        efficiency = base_efficiency + (month_idx * 1.5) + np.random.normal(0, 2)
        efficiency = min(99.9, max(50.0, efficiency))
        
        data.append({
            "branch_name": b["name"],
            "region": b["region"],
            "month": month,
            "lat": b["lat"],
            "lon": b["lon"],
            "staff_count": staff,
            "daily_transactions": transactions,
            "customer_satisfaction": satisfaction,
            "efficiency_score": round(efficiency, 2)
        })

df = pd.DataFrame(data)
df.to_csv("branch_data.csv", index=False)
print(f"[SUCCESS] Created historical branch_data.csv with {len(df)} records.")

# Write directly to SQLite database for zero-friction setup
import sqlite3
with sqlite3.connect("bharat_banking.db") as conn:
    df.to_sql("branch_performance", conn, if_exists="replace", index=False)
print(f"[SUCCESS] Populated bharat_banking.db with {len(df)} records in table 'branch_performance'.")

# 2. Train Scikit-Learn Model
print("Training AI Predictor Model...")
X = df[["staff_count", "daily_transactions", "customer_satisfaction"]]
y = df["efficiency_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)
print(f"[METRICS] Test Mean Squared Error: {mse:.4f} | R2 Score: {r2:.4f}")

# Save the model
with open("efficiency_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("[SUCCESS] Advanced ML Model saved to efficiency_model.pkl.")
