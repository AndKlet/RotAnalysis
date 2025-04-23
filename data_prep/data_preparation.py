import pandas as pd

file_path = "Skogskade_Data.csv"  


df = pd.read_csv(file_path)

# Get unique municipalities
unique_municipalities = df["Municipality"].unique()

# Dictionary to store manual location inputs
location_data = {}

print("Please enter latitude and longitude for each municipality in the format: lat,long")

for municipality in unique_municipalities:
    while True:
        user_input = input(f"Enter coordinates for {municipality} (format: lat,long): ")
        try:
            lat, lon = map(float, user_input.split(","))
            location_data[municipality] = (lat, lon)
            break  # Exit loop if input is valid
        except ValueError:
            print("Invalid format. Please enter the data in 'lat,long' format.")

# Convert the dictionary into a DataFrame
location_df = pd.DataFrame.from_dict(location_data, orient='index', columns=["Latitude", "Longitude"])
location_df.index.name = "Municipality"

# Merge with the original dataset
df = df.merge(location_df, on="Municipality", how="left")

# Save the updated dataset
output_path = "updated_file_with_locations.csv"
df.to_csv(output_path, index=False)

print(f"Updated dataset saved as {output_path}")
