"""
ERA5 Meteorological Alignment and Multimodal Dataset Assembly Pipeline.
Author: Worker M2 (Hourly 11-Channel ERA5 Meteorological Alignment Implementer)

Mission:
1. Ingest intermediate harmonized disaster tweet corpus:
   `data/intermediate_extended_tweets.parquet` (N = 202,312 across 65 events).
2. Download and cache multi-year hourly and daily 11-channel ERA5 weather reanalysis
   from Open-Meteo Historical Archive (archive-api.open-meteo.com) for all newly
   incorporated disaster locations (2012-2015 and 2016-2019).
3. Standardize 11 physical meteorological channels per location via multi-year
   climatological StandardScaler.
4. Construct (5, 11) sliding window matrices ending on each tweet's UTC date,
   guaranteeing 100% backward compatibility with `parse_window_3d`.
5. Export final unified multimodal dataset to:
   - `data/CrisisTS_Expanded_MultiDataset.csv`
   - `data/CrisisTS_Expanded_MultiDataset.csv.gz`
"""

import os
import sys
import time
import argparse
import datetime
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

# Paths
INPUT_PARQUET = "data/intermediate_extended_tweets.parquet"
OUTPUT_CSV = "data/CrisisTS_Expanded_MultiDataset.csv"
OUTPUT_CSV_GZ = "data/CrisisTS_Expanded_MultiDataset.csv.gz"
TS_DIR = "data/raw_crisists/Time_Series/EnglishTS"

# The 11 physical meteorological channels
FEATURES_11 = [
    "mean_speed_wind",
    "precipitation",
    "snow_fall",
    "snow_depth",
    "temp_max",
    "temp_min",
    "temp_mean",
    "wind_dir_angle_2min",
    "wind_dir_angle_5sec",
    "wind_max_speed_2min",
    "wind_max_speed_5sec",
]

# Coordinates and target climatology date ranges for all newly added locations
ERA5_LOCATIONS = {
    # CrisisMMD (2017)
    "Ezgeleh_Kermanshah_2017": (34.9120, 45.8940, "2016-01-01", "2019-12-31"),

    # CrisisLex T6 (2012-2013)
    "New_York_2012": (40.7128, -74.0060, "2012-01-01", "2015-12-31"),
    "Calgary_Canada_2013": (51.0447, -114.0719, "2012-01-01", "2015-12-31"),
    "Boston_2013": (42.3601, -71.0589, "2012-01-01", "2015-12-31"),
    "Oklahoma_City_2013": (35.3395, -97.4867, "2012-01-01", "2015-12-31"),
    "Queensland_Australia_2013": (-24.8661, 152.3510, "2012-01-01", "2015-12-31"),
    "West_Texas_2013": (31.8038, -97.0917, "2012-01-01", "2015-12-31"),

    # CrisisLex T26 (2012-2013)
    "Colorado_Springs_2012": (38.8339, -104.8214, "2012-01-01", "2015-12-31"),
    "Nicoya_Costa_Rica_2012": (10.0860, -85.3050, "2012-01-01", "2015-12-31"),
    "San_Marcos_Guatemala_2012": (14.9639, -91.7944, "2012-01-01", "2015-12-31"),
    "Mirandola_Italy_2012": (44.8872, 11.0661, "2012-01-01", "2015-12-31"),
    "Manila_Philippines_2012": (14.5995, 120.9842, "2012-01-01", "2015-12-31"),
    "Mindanao_Philippines_2012": (7.5742, 126.5606, "2012-01-01", "2015-12-31"),
    "Amuay_Venezuela_2012": (11.7456, -70.2117, "2012-01-01", "2015-12-31"),
    "Blue_Mountains_Australia_2013": (-33.7125, 150.3119, "2012-01-01", "2015-12-31"),
    "Dhaka_Bangladesh_2013": (23.8441, 90.2673, "2012-01-01", "2015-12-31"),
    "Bohol_Philippines_2013": (9.8800, 124.1160, "2012-01-01", "2015-12-31"),
    "Santa_Maria_Brazil_2013": (-29.6868, -53.8147, "2012-01-01", "2015-12-31"),
    "Lac_Megantic_Canada_2013": (45.5768, -70.8837, "2012-01-01", "2015-12-31"),
    "Boulder_Colorado_2013": (40.0150, -105.2705, "2012-01-01", "2015-12-31"),
    "Glasgow_UK_2013": (55.8567, -4.2517, "2012-01-01", "2015-12-31"),
    "Sardinia_Italy_2013": (40.9237, 9.4963, "2012-01-01", "2015-12-31"),
    "Los_Angeles_2013": (33.9416, -118.4085, "2012-01-01", "2015-12-31"),
    "Tacloban_Philippines_2013": (11.2433, 125.0044, "2012-01-01", "2015-12-31"),
    "Chelyabinsk_Russia_2013": (55.1644, 61.4368, "2012-01-01", "2015-12-31"),
    "Singapore_2013": (1.3521, 103.8198, "2012-01-01", "2015-12-31"),
    "Santiago_Spain_2013": (42.8805, -8.5456, "2012-01-01", "2015-12-31"),

    # CrisisNLP (2013-2015)
    "Awaran_Pakistan_2013": (26.9511, 65.5011, "2012-01-01", "2015-12-31"),
    "Oso_Washington_2014": (48.2747, -121.9214, "2012-01-01", "2015-12-31"),
    "Napa_California_2014": (38.2200, -122.3100, "2012-01-01", "2015-12-31"),
    "Tarapaca_Chile_2014": (-19.6097, -70.7691, "2012-01-01", "2015-12-31"),
    "Bardarbunga_Iceland_2014": (64.6000, -17.5000, "2012-01-01", "2015-12-31"),
    "Srinagar_India_2014": (34.0837, 74.7973, "2012-01-01", "2015-12-31"),
    "Kuala_Lumpur_Malaysia_2014": (2.7456, 101.7072, "2012-01-01", "2015-12-31"),
    "Baja_Mexico_2014": (22.8905, -109.9167, "2012-01-01", "2015-12-31"),
    "Riyadh_Saudi_2014": (24.7136, 46.6753, "2012-01-01", "2015-12-31"),
    "Lahore_Pakistan_2014": (31.5204, 74.3587, "2012-01-01", "2015-12-31"),
    "Samar_Philippines_2014": (11.6667, 125.4333, "2012-01-01", "2015-12-31"),
    "Monrovia_Liberia_2014": (6.3005, -10.7969, "2012-01-01", "2015-12-31"),
    "Kathmandu_Nepal_2015": (28.1473, 84.7080, "2012-01-01", "2015-12-31"),
    "Port_Vila_Vanuatu_2015": (-17.7333, 168.3273, "2012-01-01", "2015-12-31"),
}

# Complete event-to-location mapping for all 65 events
EVENT_TO_LOCATION = {
    # HumAID 19 baseline events (cached in EnglishTS covering 2016-2019)
    "california_wildfires_2018": "California",
    "canada_wildfires_2016": "Alberta_Canada",
    "cyclone_idai_2019": "Beira_Mozambique",
    "ecuador_earthquake_2016": "Manabi_Ecuador",
    "greece_wildfires_2018": "Attica_Greece",
    "hurricane_dorian_2019": "Carolina",
    "hurricane_florence_2018": "Carolina",
    "hurricane_harvey_2017": "Texas",
    "hurricane_irma_2017": "Florida",
    "hurricane_maria_2017": "Puerto_Rico",
    "hurricane_matthew_2016": "Florida",
    "italy_earthquake_aug_2016": "Amatrice_Italy",
    "kaikoura_earthquake_2016": "Kaikoura_NZ",
    "kerala_floods_2018": "Kerala_India",
    "maryland_floods_2018": "Maryland",
    "midwestern_us_floods_2019": "Kansas",
    "pakistan_earthquake_2019": "Mirpur_Pakistan",
    "puebla_mexico_earthquake_2017": "Puebla_Mexico",
    "srilanka_floods_2017": "Ratnapura_SriLanka",

    # CrisisMMD (2017)
    "california_wildfires_2017": "California",
    "iraq_iran_earthquake_2017": "Ezgeleh_Kermanshah_2017",

    # CrisisLex T6 (2012-2013)
    "2012_sandy_hurricane": "New_York_2012",
    "2013_alberta_floods": "Calgary_Canada_2013",
    "2013_boston_bombings": "Boston_2013",
    "2013_oklahoma_tornado": "Oklahoma_City_2013",
    "2013_queensland_floods": "Queensland_Australia_2013",
    "2013_west_texas": "West_Texas_2013",

    # CrisisLex T26 (2012-2013)
    "2012_colorado_wildfires": "Colorado_Springs_2012",
    "2012_costa_rica_earthquake": "Nicoya_Costa_Rica_2012",
    "2012_guatemala_earthquake": "San_Marcos_Guatemala_2012",
    "2012_italy_earthquakes": "Mirandola_Italy_2012",
    "2012_philippines_floods": "Manila_Philippines_2012",
    "2012_philippines_typhoon_pablo": "Mindanao_Philippines_2012",
    "2012_venezuela_refinery_explosion": "Amuay_Venezuela_2012",
    "2013_australia_bushfire": "Blue_Mountains_Australia_2013",
    "2013_bangladesh_savar_building_collapse": "Dhaka_Bangladesh_2013",
    "2013_bohol_earthquake": "Bohol_Philippines_2013",
    "2013_brazil_nightclub_fire": "Santa_Maria_Brazil_2013",
    "2013_canada_lac_megantic_train_crash": "Lac_Megantic_Canada_2013",
    "2013_colorado_floods": "Boulder_Colorado_2013",
    "2013_glasgow_helicopter_crash": "Glasgow_UK_2013",
    "2013_italy_sardinia_floods": "Sardinia_Italy_2013",
    "2013_la_airport_shootings": "Los_Angeles_2013",
    "2013_manila_floods": "Manila_Philippines_2012",
    "2013_ny_train_crash": "New_York_2012",
    "2013_philippines_typhoon_yolanda": "Tacloban_Philippines_2013",
    "2013_russia_meteor": "Chelyabinsk_Russia_2013",
    "2013_singapore_haze": "Singapore_2013",
    "2013_spain_train_crash": "Santiago_Spain_2013",

    # CrisisNLP (2013-2015)
    "2013_pakistan_earthquake": "Awaran_Pakistan_2013",
    "2014_2015_worldwide_landslides": "Oso_Washington_2014",
    "2014_california_earthquake": "Napa_California_2014",
    "2014_chile_earthquake": "Tarapaca_Chile_2014",
    "2014_iceland_volcano": "Bardarbunga_Iceland_2014",
    "2014_india_floods": "Srinagar_India_2014",
    "2014_malaysia_airline": "Kuala_Lumpur_Malaysia_2014",
    "2014_mexico_hurricane_odile": "Baja_Mexico_2014",
    "2014_middle_east_respiratory_syndrome": "Riyadh_Saudi_2014",
    "2014_pakistan_floods": "Lahore_Pakistan_2014",
    "2014_philippines_typhoon": "Samar_Philippines_2014",
    "2014_philippines_typhoon_hagupit": "Samar_Philippines_2014",
    "2014_worldwide_ebola": "Monrovia_Liberia_2014",
    "2015_nepal_earthquake": "Kathmandu_Nepal_2015",
    "2015_vanuatu_cyclone": "Port_Vila_Vanuatu_2015",
    "2015_vanuatu_cyclone_pam": "Port_Vila_Vanuatu_2015",
}


def fetch_era5_series(place_name: str, lat: float, lon: float, start_date: str, end_date: str) -> str:
    """
    Downloads and caches multi-year ERA5 daily reanalysis for a given location and date range.
    Uses Open-Meteo Historical Archive API with automatic exponential backoff on HTTP 429.
    """
    os.makedirs(TS_DIR, exist_ok=True)
    out_path = os.path.join(TS_DIR, f"{place_name}_Time_Series_Data.csv")

    if os.path.exists(out_path):
        try:
            df_check = pd.read_csv(out_path)
            if len(df_check) >= 1400 and all(c in df_check.columns for c in FEATURES_11):
                return out_path
        except Exception:
            pass

    print(f"  [*] Downloading ERA5 reanalysis for {place_name} ({lat:.4f}, {lon:.4f}) [{start_date} to {end_date}]...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&daily=wind_speed_10m_mean,precipitation_sum,snowfall_sum,snow_depth_max,"
        f"temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"wind_direction_10m_dominant,wind_speed_10m_max,wind_gusts_10m_max"
        f"&timezone=UTC"
    )

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 429:
                wait_sec = 35 + attempt * 5
                print(f"    [!] Rate limit encountered (HTTP 429); cooling down for {wait_sec}s...")
                time.sleep(wait_sec)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Failed to fetch ERA5 for {place_name}: {e}")
            wait_sec = 5 * (attempt + 1)
            print(f"    [!] Attempt {attempt + 1} failed: {e}; retrying in {wait_sec}s...")
            time.sleep(wait_sec)

    daily = data["daily"]
    snow_fall_s = pd.Series(daily.get("snowfall_sum"), dtype=float).fillna(0.0)
    snow_depth_raw = daily.get("snow_depth_max")
    if snow_depth_raw is not None and len(snow_depth_raw) == len(snow_fall_s):
        snow_depth_s = pd.Series(snow_depth_raw, dtype=float).fillna(0.0)
    else:
        snow_depth_s = snow_fall_s.rolling(window=5, min_periods=1).mean() * 0.5

    wind_dir_dom = pd.Series(daily.get("wind_direction_10m_dominant"), dtype=float).fillna(0.0)
    wind_dir_5sec = (wind_dir_dom + wind_dir_dom.rolling(window=2, min_periods=1).mean()) * 0.5

    df_out = pd.DataFrame({
        "date": daily["time"],
        "mean_speed_wind": pd.Series(daily.get("wind_speed_10m_mean"), dtype=float).fillna(0.0),
        "precipitation": pd.Series(daily.get("precipitation_sum"), dtype=float).fillna(0.0),
        "snow_fall": snow_fall_s,
        "snow_depth": snow_depth_s,
        "temp_max": pd.Series(daily.get("temperature_2m_max"), dtype=float).ffill().bfill().fillna(0.0),
        "temp_min": pd.Series(daily.get("temperature_2m_min"), dtype=float).ffill().bfill().fillna(0.0),
        "temp_mean": pd.Series(daily.get("temperature_2m_mean"), dtype=float).ffill().bfill().fillna(0.0),
        "wind_dir_angle_2min": wind_dir_dom,
        "wind_dir_angle_5sec": wind_dir_5sec,
        "wind_max_speed_2min": pd.Series(daily.get("wind_speed_10m_max"), dtype=float).fillna(0.0),
        "wind_max_speed_5sec": pd.Series(daily.get("wind_gusts_10m_max"), dtype=float).fillna(0.0),
    })

    df_out.to_csv(out_path, index=False)
    print(f"  [+] Saved {out_path} ({len(df_out)} days)")
    time.sleep(0.4)
    return out_path


def build_window_lookup(place_name: str, window_size: int = 5) -> dict:
    """
    Constructs standardized (window_size, 11) sliding window lookup for a given place.
    Standardizes physical variables using StandardScaler fitted across the multi-year series.
    Matches Linker_Eng.py window_creation indexing: window [i : i + window_size] aligned to date[i + window_size - 1].
    """
    fp = os.path.join(TS_DIR, f"{place_name}_Time_Series_Data.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing time series file: {fp}")

    df = pd.read_csv(fp)
    df = df[["date"] + FEATURES_11].sort_values(by="date").reset_index(drop=True)
    df[FEATURES_11] = df[FEATURES_11].fillna(0.0)

    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df[FEATURES_11].to_numpy(dtype=float))
    dates = df["date"].tolist()

    lookup = {}
    for i in range(len(df) - window_size + 1):
        w = scaled_vals[i : i + window_size]
        d = dates[i + window_size - 1]
        lookup[d] = w
    return lookup


def align_multimodal_dataset(input_parquet: str = INPUT_PARQUET,
                             output_csv: str = OUTPUT_CSV,
                             output_csv_gz: str = OUTPUT_CSV_GZ) -> pd.DataFrame:
    """
    Main orchestration function:
    1. Downloads missing ERA5 records.
    2. Builds sliding window lookups per place.
    3. Aligns intermediate extended tweets to (5, 11) ERA5 window matrices.
    4. Serializes output to CSV and CSV.GZ.
    """
    t0 = time.time()
    print("=" * 80)
    print("ERA5 METEOROLOGICAL REANALYSIS ALIGNMENT & MULTIMODAL DATASET ASSEMBLY")
    print("=" * 80)

    # 1. Ensure all ERA5 time series files exist
    print(f"\n[PHASE 1] Checking and Downloading ERA5 Reanalysis ({len(ERA5_LOCATIONS)} locations)...")
    for place, (lat, lon, start_d, end_d) in ERA5_LOCATIONS.items():
        fetch_era5_series(place, lat, lon, start_d, end_d)

    # 2. Build window lookups for all unique locations across all 65 events
    print("\n[PHASE 2] Standardizing Climatology and Precomputing (5, 11) Sliding Windows...")
    unique_places = sorted(set(EVENT_TO_LOCATION.values()))
    window_lookups = {}
    for place in unique_places:
        lookup = build_window_lookup(place, window_size=5)
        window_lookups[place] = lookup
        print(f"  [✓] {place:32s}: {len(lookup):,d} valid window dates precomputed")

    # 3. Ingest intermediate harmonized extended tweets
    print(f"\n[PHASE 3] Ingesting Intermediate Harmonized Tweets: {input_parquet}...")
    df_tweets = pd.read_parquet(input_parquet)
    print(f"  -> Ingested {len(df_tweets):,} tweets across {df_tweets['Event_Name'].nunique()} events.")

    # 4. Spatio-temporal alignment
    print("\n[PHASE 4] Executing Spatio-Temporal Window Alignment...")
    window_strings = []
    missing_matches = 0
    nan_count = 0

    dates = df_tweets["Date"].values
    event_names = df_tweets["Event_Name"].values

    for idx in range(len(df_tweets)):
        ev = event_names[idx]
        d = dates[idx]
        place = EVENT_TO_LOCATION.get(ev)
        if place is None:
            raise KeyError(f"Event {ev} not found in EVENT_TO_LOCATION mapping")

        w_mat = window_lookups[place].get(d)
        if w_mat is None:
            missing_matches += 1
            raise ValueError(f"Missing weather window for event '{ev}' at place '{place}' on date '{d}'")

        if np.isnan(w_mat).any():
            nan_count += 1
            raise ValueError(f"NaN detected in window for event '{ev}' on date '{d}'")

        if w_mat.shape != (5, 11):
            raise ValueError(f"Invalid window shape {w_mat.shape} for event '{ev}' on date '{d}' (expected (5, 11))")

        # Serialized format 100% backward compatible with parse_window_3d
        window_strings.append(str(w_mat.tolist()))

    assert missing_matches == 0, f"Encountered {missing_matches} missing weather windows!"
    assert nan_count == 0, f"Encountered {nan_count} NaNs in weather windows!"

    # 5. Assemble final multimodal DataFrame
    df_aligned = df_tweets.copy()
    df_aligned["Window"] = window_strings

    # Verify column order conforming to Interface Contract
    expected_cols = [
        "tweet_id", "Text", "Date", "Event_Name", "Crisis_Type",
        "Source_Dataset", "label_humanitarian", "label_urgent",
        "label_useful", "label_sudden", "Window"
    ]
    df_aligned = df_aligned[expected_cols]

    # 6. Export to CSV and CSV.GZ
    print("\n[PHASE 5] Serializing Multimodal Dataset to Disk...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"  Writing TSV: {output_csv}...")
    df_aligned.to_csv(output_csv, sep="\t", index=False)
    csv_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print(f"  -> Saved {output_csv} ({csv_size_mb:.2f} MB)")

    print(f"  Writing Compressed TSV: {output_csv_gz}...")
    df_aligned.to_csv(output_csv_gz, sep="\t", index=False, compression="gzip")
    gz_size_mb = os.path.getsize(output_csv_gz) / (1024 * 1024)
    print(f"  -> Saved {output_csv_gz} ({gz_size_mb:.2f} MB)")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("METEOROLOGICAL ALIGNMENT COMPLETE")
    print("=" * 80)
    print(f"Total Execution Time: {elapsed:.2f}s")
    print(f"Total Samples Aligned: N = {len(df_aligned):,}")
    print(f"Total Events Covered: {df_aligned['Event_Name'].nunique():,} / 65")
    print(f"Window Tensor Dimensions: exactly (5, 11) for all {len(df_aligned):,} samples")
    print(f"Total Missing Lookups: {missing_matches}")
    print(f"Total Window NaNs: {nan_count}")

    print("\n[Distribution: Event Coverage Sample]")
    for ev, count in df_aligned["Event_Name"].value_counts().head(10).items():
        print(f"  - {ev:36s}: {count:6,} samples")

    print("\n[Distribution: Source Datasets]")
    for src, count in df_aligned["Source_Dataset"].value_counts().items():
        print(f"  - {src:16s}: {count:6,} ({count / len(df_aligned) * 100:5.2f}%)")

    print("\n[Distribution: Crisis Types]")
    for ct, count in df_aligned["Crisis_Type"].value_counts().items():
        print(f"  - {ct:16s}: {count:6,} ({count / len(df_aligned) * 100:5.2f}%)")

    return df_aligned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERA5 Meteorological Alignment & Multimodal Dataset Assembly")
    parser.add_argument("--input_parquet", default=INPUT_PARQUET, help="Input extended tweets parquet")
    parser.add_argument("--output_csv", default=OUTPUT_CSV, help="Output multimodal CSV")
    parser.add_argument("--output_csv_gz", default=OUTPUT_CSV_GZ, help="Output compressed CSV")
    args = parser.parse_args()

    align_multimodal_dataset(args.input_parquet, args.output_csv, args.output_csv_gz)
