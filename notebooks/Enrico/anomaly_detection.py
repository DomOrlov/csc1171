import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from collections import Counter

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

report_folder = "anomaly_reports"
dictionary_file = "company_dictionary.csv"  # Company to Sector mapping



def parse_anomaly_file(file_path):
    # Extract dates of the anomalies for each company report
    anomalies = {}
    current_method = None
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("----") and "Anomalies" in line:
                current_method = line.replace("----","").replace("Anomalies","").strip()
                anomalies[current_method] = []
            elif line and current_method and not line.startswith("No anomalies"):
                try:
                    date_str = line.split("|")[0].strip()
                    date = pd.to_datetime(date_str)
                    anomalies[current_method].append(date)
                except:
                    continue
    return anomalies

def extract_patterns(dates):
    # Extract patterns for calendar anomalies
    patterns = {
        "Weekday": [],
        "Month": [],
        "TurnOfMonth": []
    }
    for date in dates:
        patterns["Weekday"].append(date.weekday())  # 0=Mon
        patterns["Month"].append(date.month)
        if date.day <= 3:
            patterns["TurnOfMonth"].append("Start")
        elif date.day >= (pd.Period(date, freq='ME').days_in_month - 2):
            patterns["TurnOfMonth"].append("End")
        else:
            patterns["TurnOfMonth"].append("Middle")
    return patterns

def load_sector_map(dictionary_file):
    # Map company to their corresponding market sector
    sector_map = {}
    with open(dictionary_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            sector = parts[0]
            companies = parts[1:]
            for c in companies:
                sector_map[c.upper()] = sector
    return sector_map

def plot_distribution(counts, labels, title, ylabel="Number of Anomalies"):
    # Create bar plot distributions
    plt.figure(figsize=(8,4))
    plt.bar(labels, [counts.get(i,0) if isinstance(i,int) else counts.get(i,0) for i in labels], color="skyblue")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# Run through all the files

sector_map = load_sector_map(dictionary_file)
sector_summary = {}

# Aggregate counts per sector and method
for file in os.listdir(report_folder):
    if not file.endswith("_anomalies.txt"):
        continue
    
    company_name = file.replace("_anomalies.txt", "").upper()
    sector = sector_map.get(company_name, "Unknown")
    
    file_path = os.path.join(report_folder, file)
    anomalies = parse_anomaly_file(file_path)
    
    for method, dates in anomalies.items():
        if not dates:
            continue
        
        patterns = extract_patterns(dates)
        
        # Initialize sector/method if not exists
        if sector not in sector_summary:
            sector_summary[sector] = {}
        if method not in sector_summary[sector]:
            sector_summary[sector][method] = {
                "Weekday": Counter(),
                "Month": Counter(),
                "TurnOfMonth": Counter(),
            }
        
        # Update counts
        sector_summary[sector][method]["Weekday"].update(patterns["Weekday"])
        sector_summary[sector][method]["Month"].update(patterns["Month"])
        sector_summary[sector][method]["TurnOfMonth"].update(patterns["TurnOfMonth"])

# Plot the distribution per method

weekday_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
month_labels = list(range(1,13))
turn_labels = ['Start','Middle','End']

for sector, methods in sector_summary.items():
    print(f"\n=== Sector: {sector} ===")
    for method, counts in methods.items():
        print(f"\n--- Method: {method} ---")
        
        # Weekday distribution
        plot_distribution(counts['Weekday'], list(range(7)), f"{sector} - {method} - Weekday Distribution")
        
        # Month distribution
        plot_distribution(counts['Month'], list(range(1,13)), f"{sector} - {method} - Month Distribution")
        
        # Turn-of-month distribution
        plot_distribution(counts['TurnOfMonth'], turn_labels, f"{sector} - {method} - Turn-of-Month Distribution")

