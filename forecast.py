import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/monthly_sales.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

train = df.iloc[:-4].copy()
test = df.iloc[-4:].copy()

history = list(train["Sales"])
predictions = []

for actual in test["Sales"]:
    prediction = sum(history[-3:]) / 3
    predictions.append(prediction)
    history.append(actual)

mae = mean_absolute_error(test["Sales"], predictions)
print(f"Holdout MAE: {mae:.2f}")

print("\nForecast comparison:")
for month, actual, pred in zip(test["Month"], test["Sales"], predictions):
    print(f"{month:%Y-%m}: actual={actual:.1f}, forecast={pred:.1f}")

history = list(df["Sales"])
future = []
last_date = df["Month"].iloc[-1]

for i in range(1, 4):
    pred = sum(history[-3:]) / 3
    future.append({
        "Month": last_date + pd.DateOffset(months=i),
        "Forecast": round(pred, 2)
    })
    history.append(pred)

future_df = pd.DataFrame(future)
print("\nNext 3 months:")
print(future_df.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.plot(df["Month"], df["Sales"], marker="o", label="Historical Sales")
plt.plot(future_df["Month"], future_df["Forecast"],
         marker="o", linestyle="--", label="Forecast")
plt.title("Monthly Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("sales_forecast.png", dpi=150)
plt.show()
