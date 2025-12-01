import os
import csv





script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "Global_Stock_Market_2008-2023")




output_folder = os.path.join(script_dir, "by_the_stock")


os.makedirs(output_folder, exist_ok=True)

tickers = ["^NYA", "^IXIC", "^FTSE", "^NSEI", "^BSESN", "^N225", "000001.SS", "^N100", "^DJI", "^GSPC", "GC=F", "CL=F"]  

for ticker in tickers:
    output_file = os.path.join(output_folder, f"{ticker}.csv")

    header_written = False  

    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in os.listdir(input_folder):
            if not filename.lower().endswith((".txt", ".csv")):
                continue

            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    continue

            
                file_header = lines[0].strip()
                if "\t" in file_header:
                    columns = file_header.split("\t")[1:] 
                else:
                    columns = file_header.split(",")[1:]

               
                if not header_written:
                    out_f.write(",".join(columns) + "\n")
                    header_written = True

              
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    if line.lstrip().startswith(ticker):
                        if "\t" in line:
                            data_columns = line.split("\t")[1:]
                        else:
                            data_columns = line.split(",")[1:]
                        out_f.write(",".join(data_columns) + "\n")

print("done")
