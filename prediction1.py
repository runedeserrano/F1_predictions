import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load FastF1 2024 Saudi Arabia GP race session
session_2024 = fastf1.get_session(2024, 2, "R")
session_2024.load()
# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Qualifying Data
qualifying_2025 = pd.DataFrame({
    "Driver": ["Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
               "Carlos Sainz", "Alexander Albon", "Charles Leclerc", "Yuki Tsunoda",
               "Lewis Hamilton", "Fernando Alonso", "Pierre Gasly", "Lance Stroll"],
    "QualifyingTime (s)": [86.204, 86.269, 86.375, 86.385,
                           86.569, 86.682, 86.754, 86.943,
                           87.006, 87.604, 87.710, 87.830]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Max Verstappen": "VER", "Lando Norris": "NOR", "Oscar Piastri": "PIA", "George Russell": "RUS",
    "Carlos Sainz": "SAI", "Alexander Albon": "ALB", "Charles Leclerc": "LEC", "Yuki Tsunoda": "TSU",
    "Lewis Hamilton": "HAM", "Fernando Alonso": "ALO", "Pierre Gasly": "GAS", "Lance Stroll": "STR"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 Qualifying Data with 2024 Race Data
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

# Use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Saudi Arabia GP Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
