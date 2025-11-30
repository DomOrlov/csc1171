import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


input_folder = "output"  # cleaned csv files
dictionary_file = "company_dictionary.csv"  # Company to sector mapping



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

def extract_patterns(df):
    # Weekday anomalies
    weekday_mean = df.groupby(df.index.weekday)['real_close'].transform('mean')
    weekday_std  = df.groupby(df.index.weekday)['real_close'].transform('std')
    df['Weekday_Anomaly'] = ((df['real_close'] - weekday_mean).abs() > 2.5*weekday_std).astype(int)

    # Turn-of-month anomalies
    df['Day'] = df.index.day
    df['TurnOfMonth'] = np.where(df['Day']<=3,'Start',np.where(df['Day']>=df.index.days_in_month-3,'End','Middle'))
    tom_mean = df.groupby('TurnOfMonth')['real_close'].transform('mean')
    tom_std  = df.groupby('TurnOfMonth')['real_close'].transform('std')
    df['TurnOfMonth_Anomaly'] = ((df['real_close'] - tom_mean).abs() > 2.5*tom_std).astype(int)

    # Month-of-year anomalies
    df_monthly = df['real_close'].resample('ME').mean().to_frame()
    df_monthly['Month'] = df_monthly.index.month
    month_mean = df_monthly.groupby('Month')['real_close'].transform('mean')
    month_std  = df_monthly.groupby('Month')['real_close'].transform('std')
    df_monthly['Month_Anomaly'] = ((df_monthly['real_close'] - month_mean).abs() > 2.5*month_std).astype(int)

    return df, df_monthly


def plot_distribution(counts, labels, title):
    # Plot anomalies distribution charts
    plt.figure(figsize=(8,4))
    plt.bar(labels, [counts.get(l,0) for l in labels], color="skyblue")
    plt.title(title)
    plt.ylabel("Number of Anomalies")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


sector_map = load_sector_map(dictionary_file)
sector_counts = {}

# Process all files
for file in os.listdir(input_folder):
    if not file.endswith("_cleaned_full.csv"):
        continue
    
    company_name = file.replace("_cleaned_full.csv","").upper()
    sector = sector_map.get(company_name, "Unknown")
    
    df = pd.read_csv(os.path.join(input_folder, file), parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    
    df, df_monthly = extract_patterns(df)
    
    if sector not in sector_counts:
        sector_counts[sector] = {
            'Weekday': Counter(),
            'TurnOfMonth': Counter(),
            'Month': Counter()
        }
    
    # Weekday anomalies
    weekday_counts = df[df['Weekday_Anomaly']==1].index.weekday
    sector_counts[sector]['Weekday'].update(weekday_counts)
    
    # Turn-of-month anomalies
    tom_counts = df[df['TurnOfMonth_Anomaly']==1]['TurnOfMonth']
    sector_counts[sector]['TurnOfMonth'].update(tom_counts)
    
    # Month-of-year anomalies
    month_counts = df_monthly[df_monthly['Month_Anomaly']==1]['Month']
    sector_counts[sector]['Month'].update(month_counts)

# Plot results for each sector
weekday_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
month_labels = list(range(1,13))
turn_labels = ['Start','Middle','End']

for sector, counts in sector_counts.items():
    print(f"\n=== Sector: {sector} ===")
    
    # Weekday
    plot_distribution(counts['Weekday'], list(range(7)), f"{sector} - Weekday Anomalies")
    
    # Month-of-year
    plot_distribution(counts['Month'], month_labels, f"{sector} - Month-of-Year Anomalies")
    
    # Turn-of-month
    plot_distribution(counts['TurnOfMonth'], turn_labels, f"{sector} - Turn-of-Month Anomalies")
