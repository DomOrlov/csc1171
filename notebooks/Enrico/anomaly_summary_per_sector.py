import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


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
        elif date.day >= (pd.Period(date, freq='M').days_in_month - 2):
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
    #
    plt.figure(figsize=(8,4))
    plt.bar(labels, [counts.get(i,0) if isinstance(i,int) else counts.get(i,0) for i in labels], color="skyblue")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# ===============================
# MAIN
# ===============================

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

# ===============================
# PLOT DISTRIBUTIONS PER SECTOR & METHOD
# ===============================

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
