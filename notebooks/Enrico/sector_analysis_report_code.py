import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


report_folder = "model_reports"      # folder with company reports
dictionary_file = "company_dictionary.csv"
output_folder = "sector_report"
os.makedirs(output_folder, exist_ok=True)

# Map the company to its corresponding market sector
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

# Extract metrics
metrics_list = []

for report_file in os.listdir(report_folder):
    if not report_file.endswith("_report.txt"):
        continue

    company_name = report_file.replace("_report.txt", "").upper()
    sector = sector_map.get(company_name, "Unknown")

    with open(os.path.join(report_folder, report_file), "r") as f:
        lines = f.readlines()

    current_freq = None
    for line in lines:
        line = line.strip()
        if line.startswith("=== Daily"):
            current_freq = "Daily"
        elif line.startswith("=== Weekly"):
            current_freq = "Weekly"
        elif line.startswith("=== Monthly"):
            current_freq = "Monthly"
        elif line.startswith("=== Yearly"):
            current_freq = "Yearly"

        elif line.startswith("Random Forest:") and current_freq:
            sign_acc = float(line.split("Sign Accuracy=")[1])
            metrics_list.append({
                "Company": company_name,
                "Sector": sector,
                "Frequency": current_freq,
                "Model": "Random Forest",
                "SignAcc": sign_acc
            })
        elif line.startswith("Linear Regression:") and current_freq:
            sign_acc = float(line.split("Sign Accuracy=")[1])
            metrics_list.append({
                "Company": company_name,
                "Sector": sector,
                "Frequency": current_freq,
                "Model": "Linear Regression",
                "SignAcc": sign_acc
            })

metrics_df = pd.DataFrame(metrics_list)

# Average the models accuracy for sector
sector_accuracy = metrics_df.groupby(["Sector","Frequency","Model"])["SignAcc"].mean().reset_index()

# Save summary table
summary_file = os.path.join(output_folder, "sector_average_sign_accuracy.csv")
sector_accuracy.to_csv(summary_file, index=False)
print(f"Saved sector average sign accuracy to {summary_file}")

# Plot average accuracy per sector
plt.figure(figsize=(12,6))
sns.barplot(data=sector_accuracy, x="Sector", y="SignAcc", hue="Model")
plt.title("Average Sign Accuracy by Sector and Model")
plt.ylabel("Sign Accuracy")
plt.ylim(0,0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
