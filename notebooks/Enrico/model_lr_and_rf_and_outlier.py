import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

folder_path = "output" # Change this to folder path containing the cleaneddata
output_folder = "model_reports" # Create folder for model's reports for the assets
outlier_folder = "outlier_report" # Create folder for outliers' reports for the assets
os.makedirs(output_folder, exist_ok=True)
os.makedirs(outlier_folder, exist_ok=True)

files = [f for f in os.listdir(folder_path) if f.endswith("_cleaned_full.csv")] # Check to only use the cleaned files

for file in files:
    # Run the model for all the cleaned files present in the folder
    company_name = file.replace("_cleaned_full.csv", "")
    print(f"\n\n=== Processing {company_name} ===")


    df = pd.read_csv(os.path.join(folder_path, file), parse_dates=["Date"])  
    df.sort_values("Date", inplace=True)  

    # Compute return
    df["Return"] = df["real_close"].pct_change()  
    df = df.dropna()  

    # Select target of the model  
    df["Target"] = (df["Return"] > 0).astype(int)  

    # Features of the model  
    df["ReturnLag1"] = df["Return"].shift(1)  
    df["real_closeLag1"] = df["real_close"].shift(1)  
    df["volumeLag1"] = df["volume"].shift(1)  
    df["DayOfYear"] = df["Date"].dt.dayofyear  
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)  
    df["IsJanuary"] = (df["Date"].dt.month == 1).astype(int)  
    df["IsWeekend"] = df["Date"].dt.weekday.isin([5,6]).astype(int)  
    df = df.dropna()  

    # Divide the file in train and test section 
    train = df[df["Date"] < "2020-01-01"]  
    test  = df[df["Date"] >= "2020-01-01"]  
    feature_cols = ["ReturnLag1", "real_closeLag1", "volumeLag1",  
                "DayOfYear", "IsMonthStart", "IsJanuary", "IsWeekend"]  
    X_train = train[feature_cols]  
    y_train = train["Target"]  
    X_test  = test[feature_cols]  
    y_test  = test["Target"]
    # Check if there is enough data for modelling
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping {company_name}: not enough data after preprocessing")
        continue  # If not, skip to next file


    # Scale the data  
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled  = scaler.transform(X_test)  

    # Random forest model  
    rf = RandomForestClassifier(n_estimators=200, random_state=42)  
    rf.fit(X_train_scaled, y_train)  
    y_pred_rf = rf.predict(X_test_scaled)  
    report_rf = classification_report(y_test, y_pred_rf)  
    cm_rf = confusion_matrix(y_test, y_pred_rf)  
    coeff_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)  

    # Logistic regression model 
    lr = LogisticRegression(max_iter=1000, random_state=42)  
    lr.fit(X_train_scaled, y_train)  
    y_pred_lr = lr.predict(X_test_scaled)  
    report_lr = classification_report(y_test, y_pred_lr)  
    cm_lr = confusion_matrix(y_test, y_pred_lr)  
    coeff_lr = pd.Series(lr.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)  

    # Save the coefficients and confusion matrixes of the models to a .txt file  
    report_file = os.path.join(output_folder, f"{company_name}_report.txt")  
    with open(report_file, "w") as f:  
        f.write(f"Random Forest Classification Report for {company_name}\n\n")  
        f.write(report_rf + "\n\n")  
        f.write("Confusion Matrix:\n")  
        f.write(str(cm_rf) + "\n\n")  
        f.write("Feature Coefficients:\n")  
        f.write(coeff_rf.to_string() + "\n\n")  
        f.write(f"Logistic Regression Classification Report for {company_name}\n\n")  
        f.write(report_lr + "\n\n")  
        f.write("Confusion Matrix:\n")  
        f.write(str(cm_lr) + "\n\n")  
        f.write("Feature Coefficients:\n")  
        f.write(coeff_lr.to_string())  
        print(f"Saved report to {report_file}")
    
    # Check for outliers on weekly, monthly and yearly base 
    outlier_results = {} 
    for freq, label in [("W", "Weekly"), ("M", "Monthly"), ("Y", "Yearly")]: 
        grouped = df.resample(freq, on="Date")["Return"].mean() 
        mean = grouped.mean() 
        std = grouped.std() 
        outliers = grouped[(grouped - mean).abs() > 3*std] 
        outlier_results[label] = outliers 
        print(f"\n{label} Outliers (3 std dev) for {company_name}:") 
        print(outliers)
        
    # Save outlier presence on a .txt file
    outlier_file = os.path.join(outlier_folder, f"{company_name}_outliers.txt") 
    with open(outlier_file, "w") as f: 
        for label, series in outlier_results.items(): 
            f.write(f"{label} Outliers (3 std dev):\n") 
            if not series.empty: 
                f.write(series.to_string() + "\n\n") 
            else: 
                f.write("No outliers detected.\n\n") 
    print(f"Saved outliers report to {outlier_file}")

    # Plot confusion matrixes for both models  
    for model_name, cm in [("RF", cm_rf), ("LR", cm_lr)]:  
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix - {company_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    # Plot the features's coefficients for the models
    for model_name, imp in [("RF", coeff_rf), ("LR", coeff_lr)]:  
        plt.figure(figsize=(8,5))
        sns.barplot(x=imp.values, y=imp.index, palette="viridis")
        plt.title(f"{model_name} Feature Importance for {company_name}")
        plt.xlabel("Importance / Coefficient")
        plt.tight_layout() 
        plt.show()
