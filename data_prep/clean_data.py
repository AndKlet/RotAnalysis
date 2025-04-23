import pandas as pd

# Load dataset
df = pd.read_csv("reduced_dataset_full.csv")

# Step 1: Split data into rot = Yes and rot = No
rot_yes_df = df[df["rot"] == "Yes"].copy()
rot_no_df = df[df["rot"] == "No"].copy()

# Step 2: Filter rot=No rows with at least 3 valid environmental values
climate_cols = [
    "mean_temp_3m", "mean_temp_1y", "mean_temp_5y",
    "min_temp", "max_temp", "humidity"
]
rot_no_df = rot_no_df[rot_no_df[climate_cols].notna().sum(axis=1) >= 3]

# Step 3: Add 'year' column from date
rot_yes_df["year"] = pd.to_datetime(rot_yes_df["date"], dayfirst=True).dt.year
rot_no_df["year"] = pd.to_datetime(rot_no_df["date"], dayfirst=True).dt.year

# Step 4: Downsample rot=No for a balanced dataset (1:2 ratio)
desired_ratio = 2
target_no_count = len(rot_yes_df) * desired_ratio  # e.g., 158 * 2 = 316

# Group rot=No for diversity, sample up to 10 per group
rot_no_grouped = rot_no_df.groupby(["municipality", "year"], group_keys=False)
rot_no_diverse_pool = rot_no_grouped.apply(lambda g: g.sample(n=min(len(g), 10), random_state=42))

# Final sample for desired ratio
rot_no_sampled = rot_no_diverse_pool.sample(n=min(len(rot_no_diverse_pool), target_no_count), random_state=42)

# Step 5: Combine and shuffle
balanced_df = pd.concat([rot_yes_df, rot_no_sampled], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop helper column
balanced_df = balanced_df.drop(columns=["year"])

# Step 6: Save to file
balanced_df.to_csv("data.csv", index=False)

# Summary
print(f"Original dataset size: {len(df)}")
print(f"Filtered rot=No size (after quality check): {len(rot_no_df)}")
print(f"Rot=Yes count: {len(rot_yes_df)}, Rot=No sampled count: {len(rot_no_sampled)}")
print(f"Final dataset size: {len(balanced_df)}")
print("Balanced dataset created successfully.")
