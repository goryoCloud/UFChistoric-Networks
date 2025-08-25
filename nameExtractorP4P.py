import pandas as pd

path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/'

# Load the CSV file
file_path = f"{path}rankings_history.csv"  # Update this if the file is in a different path
df = pd.read_csv(file_path, header=None, names=["date", "division", "fighter", "rank"], parse_dates=["date"])

# Filter only Pound-for-Pound (P4P) rankings, including Men's Pound-for-Pound
df_p4p = df[df["division"].isin(["Pound-for-Pound", "Men's Pound-for-Pound"])]

# Filter by date range
date_start = "2013-02-01"
date_end = "2021-03-31"
df_p4p = df_p4p[(df_p4p["date"] >= date_start) & (df_p4p["date"] <= date_end)]

# Exclude specific fighter names
excluded_fighters = ["Jéssica Andrade", "Valentina Shevchenko", "Rose Namajunas", "Amanda Nunes", "Miesha Tate", "Holly Holm", "Joanna Jedrzejczyk", "Ronda Rousey"]
df_p4p = df_p4p[~df_p4p["fighter"].isin(excluded_fighters)]

# Get unique fighter names
unique_fighters = df_p4p["fighter"].unique()

# Convert to DataFrame
fighters_df = pd.DataFrame(unique_fighters, columns=["Fighter"])

# Save as CSV
output_path = f"{path}p4p_unique_fighters.csv"
fighters_df.to_csv(output_path, index=False)

print(f"Unique P4P fighters saved to {output_path}")