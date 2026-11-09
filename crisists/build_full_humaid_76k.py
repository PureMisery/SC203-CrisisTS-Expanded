"""
Build the Complete 76,484-Row Multimodal Disaster Benchmark (CrisisTS-Full-76k).
Scales the dataset from N = 17,628 to N ~ 76,484 human-annotated disaster reports
across all 19 global disasters by:
1. Ingesting full HumAID corpus (QCRI/HumAID-events on HuggingFace) covering train, dev, and test.
2. Decoding exact UTC dates from 64-bit Twitter Snowflake IDs:
   timestamp_ms = (tweet_id >> 22) + 1288834974657
3. Aligning with cached 4-year (2016-2019) daily 11-channel ERA5 reanalysis time series.
4. Enforcing strict zero-duplicate filtering on (tweet_id) and (date, clean_text).
"""
import os
import sys
import datetime
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

TS_DIR = "data/raw_crisists/Time_Series/EnglishTS"
OUT_CSV = "data/CrisisTS_Full_76k.csv"

FEATURES_11 = [
    "mean_speed_wind", "precipitation", "snow_fall", "snow_depth",
    "temp_max", "temp_min", "temp_mean", "wind_dir_angle_2min",
    "wind_dir_angle_5sec", "wind_max_speed_2min", "wind_max_speed_5sec"
]

EVENT_KNOWLEDGE = {
    "california_wildfires_2018": ("Wildfire", "California"),
    "canada_wildfires_2016": ("Wildfire", "Alberta_Canada"),
    "cyclone_idai_2019": ("Hurricane", "Beira_Mozambique"),
    "ecuador_earthquake_2016": ("Earthquake", "Manabi_Ecuador"),
    "greece_wildfires_2018": ("Wildfire", "Attica_Greece"),
    "hurricane_dorian_2019": ("Hurricane", "Carolina"),
    "hurricane_florence_2018": ("Hurricane", "Carolina"),
    "hurricane_harvey_2017": ("Hurricane", "Texas"),
    "hurricane_irma_2017": ("Hurricane", "Florida"),
    "hurricane_maria_2017": ("Hurricane", "Puerto_Rico"),
    "hurricane_matthew_2016": ("Hurricane", "Florida"),
    "italy_earthquake_aug_2016": ("Earthquake", "Amatrice_Italy"),
    "kaikoura_earthquake_2016": ("Earthquake", "Kaikoura_NZ"),
    "kerala_floods_2018": ("Flood", "Kerala_India"),
    "maryland_floods_2018": ("Flood", "Maryland"),
    "midwestern_us_floods_2019": ("Flood", "Kansas"),
    "pakistan_earthquake_2019": ("Earthquake", "Mirpur_Pakistan"),
    "puebla_mexico_earthquake_2017": ("Earthquake", "Puebla_Mexico"),
    "srilanka_floods_2017": ("Flood", "Ratnapura_SriLanka"),
}


def decode_snowflake_date(tid_val):
    try:
        tid = int(tid_val)
        ts_ms = (tid >> 22) + 1288834974657
        dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def build_window_lookup(place: str, window_size: int = 5):
    fp = os.path.join(TS_DIR, f"{place}_Time_Series_Data.csv")
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


def map_labels(humaid_class: str, crisis_type: str):
    if humaid_class == "not_humanitarian":
        lbl_useful = "not_humanitarian"
        lbl_urgent = "not_humanitarian"
        lbl_sudden = "Not_Crisis_period"
    else:
        lbl_useful = "useful"
        lbl_urgent = "not_urgent" if humaid_class == "rescue_volunteering_or_donation_effort" else "urgent"
        lbl_sudden = "Sudden_Crisis" if crisis_type == "Earthquake" else "Non_Sudden_Crisis"
    return lbl_useful, lbl_urgent, lbl_sudden


def main():
    print("=" * 80)
    print("EXPANDING CRISISTS BENCHMARK TO FULL HUMAID 76K MULTIMODAL DATASET")
    print("=" * 80)

    # 1. Load ERA5 window lookups
    print("Loading 4-year ERA5 weather window lookups...")
    places = sorted(set(p for _, p in EVENT_KNOWLEDGE.values()))
    window_lookups = {p: build_window_lookup(p) for p in places}
    print(f"Loaded lookups for {len(places)} global impact regions.")

    # 2. Query HuggingFace for all 19 event parquets
    print("\nFetching HuggingFace QCRI/HumAID-events metadata...")
    r = requests.get('https://huggingface.co/api/datasets/QCRI/HumAID-events/parquet', timeout=15)
    if r.status_code != 200:
        print(f"Error fetching parquet list: HTTP {r.status_code}", file=sys.stderr)
        sys.exit(1)
    parquet_map = r.json()

    seen_tweet_ids = set()
    seen_date_text = set()
    rows = []
    missing_weather_count = 0

    print("\nProcessing all 19 disaster events (train + dev + test)...")
    for event_name in sorted(EVENT_KNOWLEDGE.keys()):
        crisis_type, place = EVENT_KNOWLEDGE[event_name]
        splits = parquet_map.get(event_name, {})
        ev_count = 0

        for split_name in ["train", "dev", "test"]:
            urls = splits.get(split_name, [])
            if not urls:
                continue
            url = urls[0]
            try:
                df_split = pd.read_parquet(url)
            except Exception as e:
                print(f"  Warning: Failed to read {event_name}/{split_name}: {e}")
                continue

            for _, rec in df_split.iterrows():
                tid = str(rec["tweet_id"])
                if tid in seen_tweet_ids:
                    continue

                date_str = decode_snowflake_date(tid)
                if not date_str:
                    continue

                raw_text = str(rec.get("tweet_text", "")).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
                if not raw_text:
                    continue

                dt_key = (date_str, raw_text)
                if dt_key in seen_date_text:
                    continue

                # Weather lookup
                w_mat = window_lookups[place].get(date_str)
                if w_mat is None:
                    missing_weather_count += 1
                    continue

                seen_tweet_ids.add(tid)
                seen_date_text.add(dt_key)

                humaid_class = str(rec.get("class_label", "")).strip()
                lbl_useful, lbl_urgent, lbl_sudden = map_labels(humaid_class, crisis_type)
                w_str = str(w_mat.tolist())

                rows.append({
                    "Text": raw_text,
                    "Window": w_str,
                    "label_useful": lbl_useful,
                    "label_urgent": lbl_urgent,
                    "label_humanitarian": humaid_class,
                    "label_sudden": lbl_sudden,
                    "Crisis_Type": crisis_type,
                    "Event_Name": event_name,
                })
                ev_count += 1

        print(f"  [+] {event_name:30s} ({crisis_type:10s}): {ev_count:,} unique rows aligned")

    df_full = pd.DataFrame(rows)
    print(f"\n================================================================================")
    print(f"DATASET EXPANSION COMPLETE")
    print(f"Total Unique Multimodal Samples: N = {len(df_full):,}")
    print(f"Breakdown by Crisis Type:")
    for ct, count in df_full["Crisis_Type"].value_counts().items():
        print(f"  - {ct:12s}: {count:,} ({count/len(df_full)*100:.1f}%)")
    print(f"Breakdown by Humanitarian Class (10 classes):")
    for hc, count in df_full["label_humanitarian"].value_counts().items():
        print(f"  - {hc:38s}: {count:,} ({count/len(df_full)*100:.1f}%)")
    print(f"Unmatched dates: {missing_weather_count:,}")

    # Save to disk
    df_full.to_csv(OUT_CSV, sep="\t", index=False)
    print(f"\nSaved full dataset to {OUT_CSV} ({os.path.getsize(OUT_CSV) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
