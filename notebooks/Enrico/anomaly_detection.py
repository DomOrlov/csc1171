import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

input_folder = "output"        # Folder containing the cleaned csv files


# Plot for better visualization
def show_anomaly_plot(df, col, anomaly_mask, title):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df[col], label=col, alpha=0.8)

    # Highlight anomalies in red
    plt.scatter(
        df.index[anomaly_mask == 1],
        df[col][anomaly_mask == 1],
        color="red",
        s=50,
        label="Anomaly"
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Process all files
files = [f for f in os.listdir(input_folder) if f.endswith("_cleaned_full.csv")]

for file in files:
    asset = file.replace("_cleaned_full.csv", "")
    print(f"\n=== Processing {asset} ===")

    # Open file
    df = pd.read_csv(os.path.join(input_folder, file), parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Use only essential columns
    df = df[["real_close", "volume", "close"]].dropna()

    # Sample data monthly
    df_monthly = df.resample("M").mean().copy()

    # Isolation forest anomalies
    iso = IsolationForest(contamination=0.03, random_state=42)
    df_monthly["IF_pred"] = iso.fit_predict(df_monthly[["real_close"]])
    df_monthly["IF_anomaly"] = (df_monthly["IF_pred"] == -1).astype(int)

    show_anomaly_plot(
        df_monthly,
        "real_close",
        df_monthly["IF_anomaly"],
        f"{asset} — Isolation Forest Monthly Anomalies"
    )

    # 3 std deviation monthly anomalies
    mean_m = df_monthly["real_close"].mean()
    std_m = df_monthly["real_close"].std()

    df_monthly["Sigma_anomaly"] = (
        (df_monthly["real_close"] - mean_m).abs() > 3 * std_m
    ).astype(int)

    show_anomaly_plot(
        df_monthly,
        "real_close",
        df_monthly["Sigma_anomaly"],
        f"{asset} — 3-Sigma Monthly Anomalies"
    )

    # Linear regression anomalies

    # Use only months with non-NaN values
    df_reg = df_monthly[["real_close"]].dropna().copy()

    df_monthly["Reg_anomaly"] = 0
    df_monthly["Reg_resid"] = np.nan

    if len(df_reg) > 2:

        # Fit regression
        lr = LinearRegression()
        idx = np.arange(len(df_reg)).reshape(-1, 1)
        lr.fit(idx, df_reg["real_close"])

        # Predict + compute residuals on df_reg
        predicted = lr.predict(idx)
        residuals = df_reg["real_close"] - predicted

        # Assign residuals back to df_monthly on correct dates
        df_monthly.loc[df_reg.index, "Reg_resid"] = residuals

        # Compute anomaly threshold
        resid_std = residuals.std()

        # Mark anomalies only for rows that participated in regression
        df_monthly.loc[df_reg.index, "Reg_anomaly"] = (
            residuals.abs() > 3 * resid_std).astype(int)

    else:
        # Not enough data 
        df_monthly["Reg_anomaly"] = 0


    # Seasonal anomalies
    df_monthly["Month"] = df_monthly.index.month

    seasonal_mean = df_monthly.groupby("Month")["real_close"].transform("mean")
    seasonal_std  = df_monthly.groupby("Month")["real_close"].transform("std")

    df_monthly["Seasonal_anomaly"] = (
        (df_monthly["real_close"] - seasonal_mean).abs() > 2.5 * seasonal_std
    ).astype(int)

    show_anomaly_plot(
        df_monthly,
        "real_close",
        df_monthly["Seasonal_anomaly"],
        f"{asset} — Seasonal Monthly Anomalies"
    )

    # 3 std deviation yearly anomalies
    df_yearly = df.resample("Y").mean().copy()

    mean_y = df_yearly["real_close"].mean()
    std_y  = df_yearly["real_close"].std()

    df_yearly["Yearly_anomaly"] = (
        (df_yearly["real_close"] - mean_y).abs() > 3 * std_y
    ).astype(int)

    show_anomaly_plot(
        df_yearly,
        "real_close",
        df_yearly["Yearly_anomaly"],
        f"{asset} — Yearly 3 Sigma Anomalies"
    )

    print(f"Displayed all anomaly plots for: {asset}")
    
    # Save the anomalies for each methodology

    report_folder = "anomaly_reports"
    os.makedirs(report_folder, exist_ok=True)

    report_path = os.path.join(report_folder, f"{asset}_anomalies.txt")

    with open(report_path, "w") as f:

        f.write(f"=== ANOMALY REPORT FOR {asset} ===\n\n")

    # Isolation forest anomalies
        f.write("---- Isolation Forest Monthly Anomalies ----\n")
        if df_monthly["IF_anomaly"].sum() > 0:
            for date, row in df_monthly[df_monthly["IF_anomaly"] == 1].iterrows():
                f.write(f"{date.date()}  | real_close={row['real_close']:.4f}\n")
        else:
            f.write("No anomalies detected.\n")
        f.write("\n")

    # Monthly 3 sigma anomalies
        f.write("---- 3-Sigma Monthly Anomalies ----\n")
        if df_monthly["Sigma_anomaly"].sum() > 0:
            for date, row in df_monthly[df_monthly["Sigma_anomaly"] == 1].iterrows():
                f.write(f"{date.date()}  | real_close={row['real_close']:.4f}\n")
        else:
            f.write("No anomalies detected.\n")
        f.write("\n")

    # Monthly regression anomalies
        f.write("---- Regression Monthly Anomalies ----\n")
        if df_monthly["Reg_anomaly"].sum() > 0:
            for date, row in df_monthly[df_monthly["Reg_anomaly"] == 1].iterrows():
                f.write(
                    f"{date.date()}  | real_close={row['real_close']:.4f}  "
                    f"| resid={row['Reg_resid']:.4f}\n")
        else:
            f.write("No anomalies detected.\n")
        f.write("\n")

    # Seasonal anomalies
        f.write("---- Seasonal Monthly Anomalies ----\n")
        if df_monthly["Seasonal_anomaly"].sum() > 0:
            for date, row in df_monthly[df_monthly["Seasonal_anomaly"] == 1].iterrows():
                f.write(f"{date.date()}  | real_close={row['real_close']:.4f}\n")
        else:
            f.write("No seasonal anomalies.\n")
        f.write("\n")

    # Yearly 3 sigma anomalies
        f.write("---- Yearly 3-Sigma Anomalies ----\n")
        if df_yearly["Yearly_anomaly"].sum() > 0:
            for date, row in df_yearly[df_yearly["Yearly_anomaly"] == 1].iterrows():
                f.write(f"{date.date()}  | real_close={row['real_close']:.4f}\n")
        else:
            f.write("No yearly anomalies.\n")
        f.write("\n")

    # Summary
        f.write("==== SUMMARY ====\n")
        f.write(f"Isolation Forest anomalies: {df_monthly['IF_anomaly'].sum()}\n")
        f.write(f"3-Sigma monthly anomalies: {df_monthly['Sigma_anomaly'].sum()}\n")
        f.write(f"Regression anomalies: {df_monthly['Reg_anomaly'].sum()}\n")
        f.write(f"Seasonal anomalies: {df_monthly['Seasonal_anomaly'].sum()}\n")
        f.write(f"3-Sigma yearly anomalies: {df_yearly['Yearly_anomaly'].sum()}\n")
        f.write("\n")

    print(f"Saved anomaly report → {report_path}")


print("\n=== DONE — all anomaly plots displayed in console ===")
