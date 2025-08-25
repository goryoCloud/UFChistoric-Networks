import pandas as pd

path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/'

# Load the CSV file
file_path = f"{path}rankings_history.csv" # Update this if the file is in a different path
df = pd.read_csv(file_path, header=None, names=["date", "division", "fighter", "rank"], parse_dates=["date"])

# Ensure rank column is numeric
df["rank"] = pd.to_numeric(df["rank"], errors='coerce')

# Ensure date column is in datetime format
df["date"] = pd.to_datetime(df["date"], errors='coerce')

# Filter only fighters ranked 0 from specified categories
categories = ["Heavyweight", "Light Heavyweight", "Middleweight", "Welterweight", "Lightweight", "Featherweight", "Bantamweight", "Flyweight"]
date_start = pd.to_datetime("2013-02-01")
date_end = pd.to_datetime("2021-03-31")
df_ranked = df[(df["division"].isin(categories)) & (df["rank"] == 0) & (df["date"].notna()) & (df["date"] >= date_start) & (df["date"] <= date_end)]

# Get unique fighter names
ranked_fighters = df_ranked["fighter"].unique()

# Convert to DataFrame
ranked_fighters_df = pd.DataFrame(ranked_fighters, columns=["Fighter"])

output_path_ranked = f"{path}ranked0_unique_fighters.csv"
ranked_fighters_df.to_csv(output_path_ranked, index=False)

print(f"Ranked fighters saved to {output_path_ranked}")