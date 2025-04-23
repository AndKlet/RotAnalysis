import requests
import pandas as pd
from math import radians, cos, sin, sqrt, atan2
from datetime import datetime, timedelta

CLIENT_ID = ""

# Create a requests session for efficiency
session = requests.Session()
session.auth = (CLIENT_ID, "")

# Load the existing dataset
input_csv_path = "/Users/andklet/Documents/Projects/RotDataset/reduced_dataset.csv"
df = pd.read_csv(input_csv_path)

# Function to calculate Haversine distance between two latitude/longitude points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Fetch all available weather stations
def fetch_all_stations():
    stations_endpoint = "https://frost.met.no/sources/v0.jsonld"
    print("Fetching all available weather stations...")
    
    try:
        response = session.get(stations_endpoint)
        data = response.json()
    except Exception as e:
        print(f"Error fetching station list: {e}")
        return []

    if response.status_code != 200 or "data" not in data:
        print("Failed to retrieve station list.")
        return []

    stations = []
    for station in data["data"]:
        if "geometry" in station and "coordinates" in station["geometry"]:
            lon, lat = station["geometry"]["coordinates"]
            station_id = station["id"]
            stations.append((station_id, lat, lon))
    
    print(f"Retrieved {len(stations)} weather stations.")
    return stations

# Find all stations within a given radius
def get_nearby_stations(lat, lon, stations, max_distance_km=20):
    print(f"Finding stations within {max_distance_km} km of ({lat}, {lon})...")
    nearby_stations = []
    for station_id, station_lat, station_lon in stations:
        distance = haversine(lat, lon, station_lat, station_lon)
        if distance <= max_distance_km:
            nearby_stations.append(station_id)
    
    print(f"Found {len(nearby_stations)} stations nearby: {nearby_stations}")
    return nearby_stations

# Get available elements for a station
def get_available_elements(station_id):
    test_endpoint = "https://frost.met.no/observations/v0.jsonld"
    params = {
        "sources": station_id,
        "referencetime": "2024-01-01/2024-01-02",
        "elements": "mean(air_temperature P1D),min(air_temperature P1D),max(air_temperature P1D),mean(relative_humidity P1D),volume_fraction_of_water_in_soil SW10"
    }
    
    try:
        response = session.get(test_endpoint, params=params)
        data = response.json()
        if response.status_code == 200 and "data" in data:
            observed_elements = set()
            for record in data["data"]:
                for obs in record.get("observations", []):
                    observed_elements.add(obs["elementId"])
            print(f"Station {station_id} has elements: {observed_elements}")
            return list(observed_elements)
    except Exception as e:
        print(f"Error fetching test data for {station_id}: {e}")
    return []

# Get weather data for a given station and date
def get_weather_data(station_id, date_str):
    try:
        date_str = datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None

    available_elements = get_available_elements(station_id)
    if not available_elements:
        return None

    end_date = datetime.fromisoformat(date_str)
    start_date_5y = end_date - timedelta(days=5 * 365)
    start_iso = start_date_5y.date().isoformat()
    end_iso = end_date.date().isoformat()

    observations_endpoint = "https://frost.met.no/observations/v0.jsonld"
    params = {
        "sources": station_id,
        "referencetime": f"{start_iso}/{end_iso}",
        "elements": ",".join(available_elements)
    }

    print(f"Fetching weather data for {station_id} from {start_iso} to {end_iso}...")
    try:
        response = session.get(observations_endpoint, params=params)
        obs_json = response.json()
    except Exception as e:
        print(f"Error fetching observations for {station_id}: {e}")
        return None

    if response.status_code != 200 or "data" not in obs_json:
        print(f"No data available for station {station_id}.")
        return None

    data_list = obs_json["data"]
    if not data_list:
        return None

    results = {
        "mean_temp_3m": [],
        "mean_temp_1y": [],
        "mean_temp_5y": [],
        "min_temp": [],
        "max_temp": [],
        "humidity": [],
        "soil_humidity": []
    }

    for entry in data_list:
        for obs in entry.get("observations", []):
            element = obs.get("elementId")
            value = obs.get("value")
            if element == "mean(air_temperature P1D)":
                results["mean_temp_3m"].append(value)
                results["mean_temp_1y"].append(value)
                results["mean_temp_5y"].append(value)
            elif element == "min(air_temperature P1D)":
                results["min_temp"].append(value)
            elif element == "max(air_temperature P1D)":
                results["max_temp"].append(value)
            elif element == "mean(relative_humidity P1D)":
                results["humidity"].append(value)
            elif element == "volume_fraction_of_water_in_soil SW10":
                results["soil_humidity"].append(value)

    return {key: (sum(values) / len(values) if values else None) for key, values in results.items()}

# Fetch stations
stations_list = fetch_all_stations()
station_cache = {}

# Add new columns
df["mean_temp_3m"] = None
df["mean_temp_1y"] = None
df["mean_temp_5y"] = None
df["min_temp"] = None
df["max_temp"] = None
df["humidity"] = None
df["soil_humidity"] = None

# Process each row
for idx, row in df.iterrows():  # Limit to 10 rows for testing
    lat, lon, date_str = row["lat"], row["long"], row["date"]
    print(f"\nProcessing record {idx} at ({lat}, {lon}) on {date_str}...")

    coord_key = (round(lat, 4), round(lon, 4))
    if coord_key in station_cache:
        station_ids = station_cache[coord_key]
    else:
        station_ids = get_nearby_stations(lat, lon, stations_list, max_distance_km=20)
        station_cache[coord_key] = station_ids

    if not station_ids:
        print(f"No stations found near ({lat}, {lon}). Skipping record {idx}.")
        continue

    weather_data = get_weather_data(station_ids[0], date_str)  # Using closest station
    if not weather_data:
        print(f"No valid data retrieved for record {idx}.")
        continue

    for key in weather_data:
        df.at[idx, key] = weather_data[key]
        print(f"  → {key}: {weather_data[key]}")

# Save to file
output_csv_path = '/Users/andklet/Documents/Projects/RotDataset/reduced_dataset_full.csv'
df.to_csv(output_csv_path, index=False)
print(f"\nMerged data saved to {output_csv_path}")
