import os
import pandas as pd
import matplotlib.pyplot as plt


folder_path = "full_history"  # Folder containing the CSV files
output_file = "data_quality_report.txt"


with open(output_file, "w", encoding="utf-8") as f:
    f.write("Data Quality Report: Consecutive Duplicate & Missing Data Checker\n")
    f.write("=" * 70 + "\n\n")


duplicate_day_counts = {}
missing_summary = []

# Process all file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        ticker = filename.replace(".csv", "")

        try:
            # Load data
            df = pd.read_csv(
                file_path,
                dtype={
                    'volume': 'Int64',
                    'open': 'float',
                    'close': 'float',
                    'high': 'float',
                    'low': 'float',
                    'adjclose': 'float'
                },
                parse_dates=['date']
            )

            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            log_lines = []
            log_lines.append(f"Ticker: {ticker}")
            log_lines.append(f"Rows: {len(df)}")

            # Missing data check
            missing_counts = df.isna().sum()
            total_missing = missing_counts.sum()
            if total_missing > 0:
                log_lines.append(f"Missing data detected ({total_missing} total missing values):")
                log_lines.append(missing_counts[missing_counts > 0].to_string())
                missing_summary.append({
                    "Ticker": ticker,
                    "Total Missing": int(total_missing),
                    **missing_counts.to_dict()
                })
            else:
                log_lines.append("No missing data found.")

            # Check for consecutive duplicate rows 
            columns_to_compare = ['open', 'close', 'high', 'low', 'adjclose', 'volume']
            shifted = df[columns_to_compare].shift(1)
            is_duplicate = (df[columns_to_compare] == shifted).all(axis=1)

            duplicated_rows = df[is_duplicate]
            count = len(duplicated_rows)

            if count > 0:
                log_lines.append(f"Found {count} rows with identical data to the previous day.")
                # Record (month, day) for each duplicate
                for date in duplicated_rows.index:
                    month_day = (date.month, date.day)
                    duplicate_day_counts[month_day] = duplicate_day_counts.get(month_day, 0) + 1

                # Log a sample of the duplicates
                sample = duplicated_rows.head(5).copy()
                sample['prev_date'] = sample.index - pd.Timedelta(days=1)
                log_lines.append("Sample of consecutive identical rows:")
                log_lines.append(sample.to_string())
            else:
                log_lines.append("No consecutive duplicate rows found.")

            # Output to file and console 
            full_log = "\n".join(log_lines)
            separator = "\n" + "-" * 60 + "\n\n"

            print(full_log)
            print(separator)

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(full_log)
                f.write(separator)

        except Exception as e:
            error_msg = f"Error processing {ticker}: {e}"
            print(error_msg)
            print("-" * 60)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(error_msg + "\n")
                f.write("-" * 60 + "\n\n")

# Summary
if missing_summary:
    print("\n Final Summary: Missing Data per Ticker")
    missing_df = pd.DataFrame(missing_summary)
    missing_df.fillna(0, inplace=True)
    missing_df = missing_df.sort_values(by="Total Missing", ascending=False)

    print(missing_df.to_string(index=False))
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n Final Summary: Missing Data per Ticker\n")
        f.write("-" * 60 + "\n")
        f.write(missing_df.to_string(index=False))
        f.write("\n" + "-" * 60 + "\n\n")


if duplicate_day_counts:
    print("\n Final Summary: Consecutive Duplicate Frequency by Day of Year")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n Final Summary: Consecutive Duplicate Frequency by Day of Year\n")
        f.write("-" * 60 + "\n")

    # Create summary DataFrame
    summary_df = pd.DataFrame(list(duplicate_day_counts.items()), columns=["month_day", "count"])
    summary_df[['month', 'day']] = pd.DataFrame(summary_df['month_day'].tolist(), index=summary_df.index)
    summary_df.drop(columns="month_day", inplace=True)
    summary_df['Date'] = summary_df['month'].astype(str).str.zfill(2) + "-" + summary_df['day'].astype(str).str.zfill(2)
    summary_df = summary_df[['month', 'day', 'Date', 'count']].rename(columns={"count": "Duplicate Count"})
    summary_df.sort_values(by=["month", "day"], inplace=True)

    # Write table to report
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(summary_df[['Date', 'Duplicate Count']].to_string(index=False))
        f.write("\n" + "-" * 60 + "\n\n")

    # Ploths per month
    months = summary_df['month'].unique()
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    for m in months:
        monthly_df = summary_df[summary_df['month'] == m]
        plt.figure(figsize=(12, 5))
        plt.bar(monthly_df['day'].astype(str), monthly_df['Duplicate Count'], color='tomato')
        plt.title(f"Consecutive Duplicate Frequency – {month_names.get(m, f'Month {m}')}")
        plt.xlabel("Day of Month")
        plt.ylabel("Number of Duplicates")
        plt.xticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()

else:
    print("No consecutive-day duplicates found in any file.")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("No consecutive-day duplicates found in any file.\n")


