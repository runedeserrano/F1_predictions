import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import getQualifiers

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")
race_name = "Monaco"

# Load 2024 race session
session_2024 = fastf1.get_session(2024, race_name, "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# getting 2025 qualifying times
quali_results = getQualifiers.get_quali_results(2025, race_name)

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "VER": "Max Verstappen", "NOR": "Lando Norris", "ANT": "Andrea Kimi Antonelli", "PIA": "Oscar Piastri", "RUS": "George Russell",
    "SAI": "Carlos Sainz", "ALB": "Alexander Albon", "LEC": "Charles Leclerc", "OCO": "Esteban Ocon", "TSU": "Yuki Tsunoda",
    "HAD": "Isack Hadjar", "HAM": "Lewis Hamilton", "BOR": "Gabriel Bortoleto", "DOO": "Jack Doohan", "LAW": "Liam Lawson",
    "HUL": "Nico Hülkenberg", "ALO": "Fernando Alonso", "GAS": "Pierre Gasly", "STR": "Lance Stroll", "BEA": "Oliver Bearman"
}

quali_results["DriverCode"] = quali_results["Driver"]
quali_results["QualifyingTime (s)"] = quali_results["LapTime"]
quali_results = quali_results[["Driver", "QualifyingTime (s)", "DriverCode"]].copy()
quali_results["Driver"] = quali_results["DriverCode"].map(driver_mapping)

# Merge 2025 Qualifying Data with 2024 Race Data
merged_data = quali_results.merge(laps_2024, left_on="DriverCode", right_on="Driver")
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
predicted_lap_times = model.predict(quali_results[["QualifyingTime (s)"]])
quali_results["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = quali_results.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 " + race_name + " Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
