import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


folder_path = "output"          # folder containing cleaned CSVs
output_folder = "model_reports" # save model reports
outlier_folder = "outlier_report"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(outlier_folder, exist_ok=True)

files = [f for f in os.listdir(folder_path) if f.endswith("_cleaned_full.csv")]

# Plots for better visualization
def plot_actual_vs_predicted(dates, y_true, y_pred, title):
    plt.figure(figsize=(14,5))
    plt.plot(dates, y_true, label='Actual Return', alpha=0.7)
    plt.plot(dates, y_pred, label='Predicted Return', alpha=0.7)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_importance(imp, feature_names, model_name, company_name):
    plt.figure(figsize=(8,5))
    sns.barplot(x=imp, y=feature_names, palette="viridis")
    plt.title(f"{model_name} Feature Importance for {company_name}")
    plt.xlabel("Importance / Coefficient")
    plt.tight_layout()
    plt.show()

def evaluate_aggregated(df_dates, y_true, y_pred, freq="W"):
# Evaluate the model against the actual data
    pred_series = pd.Series(y_pred, index=df_dates)
    true_series = pd.Series(y_true, index=df_dates)

    # Resample and compute mean returns
    true_avg = true_series.resample(freq).mean()
    pred_avg = pred_series.resample(freq).mean()

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(true_avg, pred_avg))
    r2 = r2_score(true_avg, pred_avg)
    sign_acc = np.mean((pred_avg > 0) == (true_avg > 0))

    return true_avg, pred_avg, rmse, r2, sign_acc

# Loop through the various files
for file in files:
    company_name = file.replace("_cleaned_full.csv", "")
    print(f"\n\n=== Processing {company_name} ===")

    # Load and preprocess
    df = pd.read_csv(os.path.join(folder_path, file), parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    # Compute return
    df["Return"] = df["real_close"].pct_change()
    df = df.dropna()

    # Target = continuous return
    df["Target"] = df["Return"]

    # Features
    df["ReturnLag1"] = df["Return"].shift(1)
    df["real_closeLag1"] = df["real_close"].shift(1)
    df["volumeLag1"] = df["volume"].shift(1)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsJanuary"] = (df["Date"].dt.month == 1).astype(int)
    df["IsWeekend"] = df["Date"].dt.weekday.isin([5,6]).astype(int)
    df = df.dropna()

    # Split datasets in train (2008 to 2020) to test (post 2020)
    train = df[(df["Date"] >= "2008-01-01") & (df["Date"] < "2020-01-01")]
    test  = df[df["Date"] >= "2020-01-01"]

    feature_cols = ["ReturnLag1", "real_closeLag1", "volumeLag1",
                    "DayOfYear", "IsMonthStart", "IsJanuary", "IsWeekend"]
    X_train = train[feature_cols]
    y_train = train["Target"]
    X_test = test[feature_cols]
    y_test = test["Target"]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping {company_name}: not enough data")
        continue

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random forest model
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)
    sign_accuracy_rf = np.mean((y_pred_rf > 0) == (y_test.values > 0))
    coeff_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Linear regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)
    sign_accuracy_lr = np.mean((y_pred_lr > 0) == (y_test.values > 0))
    coeff_lr = pd.Series(lr.coef_, index=feature_cols).sort_values(key=abs, ascending=False)

    # Save the coefficients for the models
    report_file = os.path.join(output_folder, f"{company_name}_report.txt")
    with open(report_file, "w") as f:
        f.write(f"=== Daily Regression Metrics ===\n")
        f.write(f"Random Forest: RMSE={rf_rmse:.6f}, R²={rf_r2:.6f}, Sign Accuracy={sign_accuracy_rf:.4f}\n")
        f.write(coeff_rf.to_string() + "\n\n")
        f.write(f"Linear Regression: RMSE={lr_rmse:.6f}, R²={lr_r2:.6f}, Sign Accuracy={sign_accuracy_lr:.4f}\n")
        f.write(coeff_lr.to_string() + "\n\n")

    # plot for the daily returns and models predictions
    plot_actual_vs_predicted(test['Date'], y_test.values, y_pred_rf, f"RF: Actual vs Predicted Returns — {company_name}")
    plot_actual_vs_predicted(test['Date'], y_test.values, y_pred_lr, f"LR: Actual vs Predicted Returns — {company_name}")
    plot_feature_importance(coeff_rf.values, coeff_rf.index, "RF", company_name)
    plot_feature_importance(coeff_lr.values, coeff_lr.index, "LR", company_name)

    # plots for the averaged weekly and monthly returns against prediscted values
    for freq, label in [("W", "Weekly"), ("M", "Monthly")]:
        true_avg, pred_rf_avg, rmse_rf_agg, r2_rf_agg, sign_rf_agg = evaluate_aggregated(test['Date'], y_test.values, y_pred_rf, freq=freq)
        _, pred_lr_avg, rmse_lr_agg, r2_lr_agg, sign_lr_agg = evaluate_aggregated(test['Date'], y_test.values, y_pred_lr, freq=freq)

        # Add metrics to report
        with open(report_file, "a") as f:
            f.write(f"=== {label} Aggregated Metrics ===\n")
            f.write(f"Random Forest: RMSE={rmse_rf_agg:.6f}, R²={r2_rf_agg:.6f}, Sign Accuracy={sign_rf_agg:.4f}\n")
            f.write(f"Linear Regression: RMSE={rmse_lr_agg:.6f}, R²={r2_lr_agg:.6f}, Sign Accuracy={sign_lr_agg:.4f}\n\n")

        # Plot aggregated averages
        plt.figure(figsize=(14,5))
        plt.plot(true_avg.index, true_avg.values, label="Actual Avg Return")
        plt.plot(pred_rf_avg.index, pred_rf_avg.values, label="RF Predicted Avg Return")
        plt.plot(pred_lr_avg.index, pred_lr_avg.values, label="LR Predicted Avg Return")
        plt.title(f"{label} Average Returns — {company_name}")
        plt.xlabel("Date")
        plt.ylabel("Average Return")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # detect all values outside 3 std of average
    outlier_results = {}
    for freq, label in [("W", "Weekly"), ("M", "Monthly"), ("Y", "Yearly")]:
        grouped = df.resample(freq, on="Date")["Return"].mean()
        mean = grouped.mean()
        std = grouped.std()
        outliers = grouped[(grouped - mean).abs() > 3*std]
        outlier_results[label] = outliers

    outlier_file = os.path.join(outlier_folder, f"{company_name}_outliers.txt")
    with open(outlier_file, "w") as f:
        for label, series in outlier_results.items():
            f.write(f"{label} Outliers (3 std dev):\n")
            if not series.empty:
                f.write(series.to_string() + "\n\n")
            else:
                f.write("No outliers detected.\n\n")
