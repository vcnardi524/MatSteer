import pandas as pd
import os

# Directory containing the Parquet files
input_dir = "benchmarking"
output_file = "thresholds_summary.csv"

# Initialize a list to store results
results = []

# Iterate over all Parquet files in the directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".parquet"):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing {file_name}...")

        try:
            # Load the Parquet file
            df = pd.read_parquet(file_path)

            # Calculate thresholds for SA, QED, and LogP
            for column in ['sa', 'qed', 'logp']:
                if column in df.columns:
                    positive_values = df[df[column] > 0][column]  # Filter positive values
                    bottom_25 = positive_values.quantile(0.25)
                    top_25 = positive_values.quantile(0.75)
                    results.append({
                        "file": file_name,
                        "metric": column,
                        "bottom_q": bottom_25,
                        "top_q": top_25
                    })
        except Exception as e:
            print(f"Skipping {file_name} due to error: {e}")

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"Thresholds saved to {output_file}")