import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

report_folder = "model_reports" # Change this to folder path containing the reports
dictionary_file = "company_dictionary.csv" # Change this to folder path containing the company dictionary
output_folder = "sector_report" # Create folder for model's reports for the market sectors
os.makedirs(output_folder, exist_ok=True)

# Create a map for the company in each sector
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

# Collect coefficients and accuracies per each file

lr_list = []
rf_list = []
accuracy_list = []

for report_file in os.listdir(report_folder):
    if not report_file.endswith("_report.txt"):
        continue

    company_name = report_file.replace("_report.txt", "").upper()
    sector = sector_map.get(company_name, "Unknown")
    with open(os.path.join(report_folder, report_file), "r") as f:
        lines = f.readlines()

    # Extract coefficients for RF
    try:
        start_idx = lines.index("Feature Coefficients:\n") + 1
        rf_lines = []
        for line in lines[start_idx:]:
            if line.strip() == "":
                break
            rf_lines.append(line)
        for l in rf_lines:
            feat, val = l.strip().split()
            rf_list.append({"Company": company_name, "Sector": sector, "Feature": feat, "RF_Coefficients": float(val)})
    except ValueError:
        pass

    # Extract coefficients for LR
    try:
        start_idx = lines.index("Feature Coefficients:\n") + 1
        coeff_lines = []
        for line in lines[start_idx:]:
            if line.strip() == "":
                break
            coeff_lines.append(line)
        for l in coeff_lines:
            feat, val = l.strip().split()
            lr_list.append({"Company": company_name, "Sector": sector, "Feature": feat, "Coefficient": float(val)})
    except ValueError:
        pass

    # Extract RF and LR accuracies and confusion matrixes
    try:
        # Random Forest confusion matrix
        rf_idx = lines.index("Confusion Matrix:\n")
        cm_rf = [[int(x) for x in lines[rf_idx+1+i].strip('[]\n ').split()] for i in range(2)]
        acc_rf = sum(cm_rf[i][i] for i in range(len(cm_rf))) / sum(sum(row) for row in cm_rf)

        # Logistic Regression confusion matrix
        lr_idx = lines[rf_idx+7:].index("Confusion Matrix:\n") + rf_idx + 7
        cm_lr = [[int(x) for x in lines[lr_idx+1+i].strip('[]\n ').split()] for i in range(2)]
        acc_lr = sum(cm_lr[i][i] for i in range(len(cm_lr))) / sum(sum(row) for row in cm_lr)

        accuracy_list.append({"Company": company_name, "Sector": sector, "RF_Accuracy": acc_rf, "LR_Accuracy": acc_lr})
    except Exception:
        pass

# Create dataframes for sector summary
lr_df = pd.DataFrame(lr_list)
rf_df = pd.DataFrame(rf_list)
accuracy_df = pd.DataFrame(accuracy_list)

# Aggregate logistic regression coefficients
sector_coeff_summary = lr_df.groupby(["Sector", "Feature"])["Coefficient"].mean().reset_index()
# Aggregate random forest coefficients
sector_rf_summary = rf_df.groupby(["Sector", "Feature"])["RF_Coefficients"].mean().reset_index()
# Aggregate accuracy of both models
acc_sector_summary = accuracy_df.groupby("Sector")[["RF_Accuracy", "LR_Accuracy"]].mean().reset_index()

# Save coefficients per sector
for sector in sector_coeff_summary["Sector"].unique():
    coeff_df_sector = sector_coeff_summary[sector_coeff_summary["Sector"] == sector]
    rf_df_sector = sector_rf_summary[sector_rf_summary["Sector"] == sector]

    txt_file = os.path.join(output_folder, f"{sector}_coefficients.txt")
    with open(txt_file, "w") as f:
        f.write(f"Average Logistic Regression Coefficients - {sector}\n")
        f.write(coeff_df_sector.to_string(index=False))
        f.write("\n\n")
        f.write(f"Average Random Forest Feature Coefficients - {sector}\n")
        f.write(rf_df_sector.to_string(index=False))
    print(f"Saved sector coefficient summary to {txt_file}")

# Save models' accuracy per sector
for sector in accuracy_df["Sector"].unique():
    df = accuracy_df[accuracy_df["Sector"] == sector]
    txt_file = os.path.join(output_folder, f"{sector}_accuracy.txt")
    with open(txt_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row['Company']}: RF Accuracy = {row['RF_Accuracy']:.4f}, LR Accuracy = {row['LR_Accuracy']:.4f}\n")
    print(f"Saved per-report accuracy report for {sector} to {txt_file}")

# Plot comparison of models accuracy for each sector
plt.figure(figsize=(10,6))
sns.barplot(data=acc_sector_summary.melt(id_vars="Sector", var_name="Model", value_name="Accuracy"),
            x="Sector", y="Accuracy", hue="Model")
plt.title("Average Model Accuracy by Sector")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

