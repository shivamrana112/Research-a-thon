import pandas as pd
import numpy as np

np.random.seed(42)
N = 1500

data = {
    "Cycle_Index": np.arange(1, N + 1),
    "Discharge_Time(s)": np.random.uniform(1200, 5000, N),
    "Decrement_3.6-3.4V(s)": np.random.uniform(40, 200, N),
    "Max_Voltage_Discharge(V)": np.random.uniform(3.85, 4.25, N),
    "Min_Voltage_Charger(V)": np.random.uniform(2.5, 3.25, N),
}

data["RUL"] = (1500 - data["Cycle_Index"]) + np.random.normal(0, 40, N)
data["RUL"] = np.clip(data["RUL"], 0, None)

df = pd.DataFrame(data)
df.to_csv("Battery_RUL.csv", index=False)

print("Battery_RUL.csv generated successfully!")
