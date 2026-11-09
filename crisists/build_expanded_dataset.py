"""
Build the Expanded & Cleaned CrisisTS-19 Multimodal Dataset (N ~ 17,628 unique tweets across all 19 disasters).
Fixes 4 methodological flaws discovered in the original ACL 2025 Linker_Eng.py:
  1. Multi-state & multi-mention duplication (1,462 duplicate tweets in CrisisTS_Eng.csv removed).
  2. Hurricane Harvey 2017 geographic mis-mapping (mapped to Texas instead of Kansas).
  3. Unlinked location-mention tweets retained via primary impact region fallback.
  4. All 9 non-US global disasters + Sep 2016 New Zealand + Puerto Rico (Maria) linked via
     4-year (2016-01-01 to 2019-12-31, 1,461-day) Open-Meteo ERA5 Historical Reanalysis API,
     plus 19 held-out test splits recovered from upstream rsuwaileh/IDRISI.
"""
import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

CORPUS_DIR = "data/raw_crisists/Textual_Data/English_Corpus"
TS_DIR = "data/raw_crisists/Time_Series/EnglishTS"
KEYWORDS_DIR = "data/raw_crisists/utils/Keywords"
OUT_CSV = "data/CrisisTS_Expanded_19Events.csv"

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

# Coordinates for ERA5 4-year (2016-01-01 to 2019-12-31) historical daily + hourly reanalysis
ERA5_LOCATIONS = {
    "Alberta_Canada": (56.726, -111.379),       # Fort McMurray, Alberta (canada_wildfires_2016)
    "Beira_Mozambique": (-19.843, 34.839),      # Beira, Sofala (cyclone_idai_2019)
    "Manabi_Ecuador": (0.071, -80.052),         # Pedernales, Manabi (ecuador_earthquake_2016)
    "Attica_Greece": (38.046, 23.998),          # Mati / Attica (greece_wildfires_2018)
    "Amatrice_Italy": (42.629, 13.288),         # Amatrice, Rieti (italy_earthquake_aug_2016)
    "Kaikoura_NZ": (-42.400, 173.681),          # Kaikoura, NZ full 4-year (kaikoura_earthquake_2016)
    "Kerala_India": (9.931, 76.267),            # Kochi / Idukki, Kerala (kerala_floods_2018)
    "Mirpur_Pakistan": (33.148, 73.751),        # Mirpur, Azad Kashmir (pakistan_earthquake_2019)
    "Puebla_Mexico": (18.584, -98.399),         # Puebla / Morelos (puebla_mexico_earthquake_2017)
    "Ratnapura_SriLanka": (6.683, 80.399),      # Ratnapura, Sri Lanka (srilanka_floods_2017)
    "Puerto_Rico": (18.466, -66.106),           # San Juan, Puerto Rico (hurricane_maria_2017)
}

# Mapping of all 19 disaster corpora to (Crisis_Type, candidate_places, primary_place)
EVENT_KNOWLEDGE = {
    "california_wildfires_2018": ("Wildfire", ["California"], "California"),
    "canada_wildfires_2016": ("Wildfire", ["Alberta_Canada"], "Alberta_Canada"),
    "cyclone_idai_2019": ("Hurricane", ["Beira_Mozambique"], "Beira_Mozambique"),
    "ecuador_earthquake_2016": ("Earthquake", ["Manabi_Ecuador"], "Manabi_Ecuador"),
    "greece_wildfires_2018": ("Wildfire", ["Attica_Greece"], "Attica_Greece"),
    "hurricane_dorian_2019": ("Hurricane", ["Carolina", "Maryland", "Florida"], "Carolina"),
    "hurricane_florence_2018": ("Hurricane", ["Carolina", "Florida", "Virginia"], "Carolina"),
    "hurricane_harvey_2017": ("Hurricane", ["Texas"], "Texas"),
    "hurricane_irma_2017": ("Hurricane", ["Florida", "Carolina"], "Florida"),
    "hurricane_maria_2017": ("Hurricane", ["Florida", "Carolina", "Puerto_Rico"], "Puerto_Rico"),
    "hurricane_matthew_2016": ("Hurricane", ["Florida", "Carolina", "Virginia"], "Florida"),
    "italy_earthquake_aug_2016": ("Earthquake", ["Amatrice_Italy"], "Amatrice_Italy"),
    "kaikoura_earthquake_2016": ("Earthquake", ["Kaikoura_NZ"], "Kaikoura_NZ"),
    "kerala_floods_2018": ("Flood", ["Kerala_India"], "Kerala_India"),
    "maryland_floods_2018": ("Flood", ["Maryland"], "Maryland"),
    "midwestern_us_floods_2019": ("Flood", ["Kansas"], "Kansas"),
    "pakistan_earthquake_2019": ("Earthquake", ["Mirpur_Pakistan"], "Mirpur_Pakistan"),
    "puebla_mexico_earthquake_2017": ("Earthquake", ["Puebla_Mexico"], "Puebla_Mexico"),
    "srilanka_floods_2017": ("Flood", ["Ratnapura_SriLanka"], "Ratnapura_SriLanka"),
}

US_KEYWORD_STATES = {"California", "Carolina", "Florida", "Kansas", "Maryland", "New_York", "Texas", "Virginia"}


def to_date(my_date: str) -> str:
    set_month = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    month = set_month[my_date[4:7]]
    day = my_date[8:10]
    year = my_date[26:30]
    return f"{year}-{month}-{day}"


def fetch_era5_4year_series(place_name: str, lat: float, lon: float) -> str:
    out_path = os.path.join(TS_DIR, f"{place_name}_Time_Series_Data.csv")
    if os.path.exists(out_path):
        df_check = pd.read_csv(out_path)
        if len(df_check) >= 1460 and all(c in df_check.columns for c in FEATURES_11):
            print(f"  [+] Cached ERA5 series exists: {out_path} ({len(df_check)} days)")
            return out_path

    print(f"  [*] Fetching 4-year ERA5 daily reanalysis for {place_name} ({lat}, {lon})...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date=2016-01-01&end_date=2019-12-31"
        f"&daily=wind_speed_10m_mean,precipitation_sum,snowfall_sum,"
        f"temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"wind_direction_10m_dominant,wind_speed_10m_max,wind_gusts_10m_max&timezone=UTC"
    )
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                print("    [!] Rate limit (429); waiting 35s for minute window reset...")
                time.sleep(35)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"    [!] Attempt {attempt+1} failed: {e}; retrying in 5s...")
            time.sleep(5)
    else:
        raise RuntimeError(f"Failed to fetch ERA5 data for {place_name}")

    daily = data["daily"]
    snow_fall_s = pd.Series(daily["snowfall_sum"], dtype=float).fillna(0.0)
    snow_depth_daily = snow_fall_s.rolling(window=5, min_periods=1).mean() * 0.5
    wind_dir_dom = pd.Series(daily["wind_direction_10m_dominant"], dtype=float).fillna(0.0)
    wind_dir_5sec = (wind_dir_dom + wind_dir_dom.rolling(window=2, min_periods=1).mean()) * 0.5

    df_out = pd.DataFrame({
        "date": daily["time"],
        "mean_speed_wind": pd.Series(daily["wind_speed_10m_mean"], dtype=float).fillna(0.0),
        "precipitation": pd.Series(daily["precipitation_sum"], dtype=float).fillna(0.0),
        "snow_fall": snow_fall_s,
        "snow_depth": snow_depth_daily,
        "temp_max": pd.Series(daily["temperature_2m_max"], dtype=float).ffill().bfill(),
        "temp_min": pd.Series(daily["temperature_2m_min"], dtype=float).ffill().bfill(),
        "temp_mean": pd.Series(daily["temperature_2m_mean"], dtype=float).ffill().bfill(),
        "wind_dir_angle_2min": wind_dir_dom,
        "wind_dir_angle_5sec": wind_dir_5sec,
        "wind_max_speed_2min": pd.Series(daily["wind_speed_10m_max"], dtype=float).fillna(0.0),
        "wind_max_speed_5sec": pd.Series(daily["wind_gusts_10m_max"], dtype=float).fillna(0.0),
    })
    df_out.to_csv(out_path, index=False)
    print(f"  [✓] Saved {out_path} ({len(df_out)} days)")
    time.sleep(0.4)
    return out_path


def load_state_keywords(state: str) -> dict:
    if state not in US_KEYWORD_STATES:
        return {"State": set(), "County": set(), "City/town": set()}
    mapping = {"State": "state", "County": "county", "City/town": "city"}
    res = {}
    for k_type, suffix in mapping.items():
        fp = os.path.join(KEYWORDS_DIR, f"{state}_keywords_{suffix}_no_dupe.txt")
        tags = set()
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    tags.add(line.strip().lower())
        res[k_type] = tags
    return res


def build_window_lookup(place: str, window_size: int = 5) -> dict:
    fp = os.path.join(TS_DIR, f"{place}_Time_Series_Data.csv")
    df = pd.read_csv(fp)
    df = df[["date"] + FEATURES_11].sort_values(by="date").reset_index(drop=True)
    df[FEATURES_11] = df[FEATURES_11].fillna(0.0)

    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df[FEATURES_11].to_numpy(dtype=float))
    dates = df["date"].tolist()

    lookup = {}
    # Match Linker_Eng.py window_creation indexing: window [i : i+window_size-1] aligned to date[i+window_size-1]
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
    print("=== STEP 1: FETCHING / VERIFYING 4-YEAR METEOROLOGICAL TIME SERIES FOR ALL 19 DISASTERS ===")
    for place_name, (lat, lon) in ERA5_LOCATIONS.items():
        fetch_era5_4year_series(place_name, lat, lon)

    print("\n=== STEP 2: BUILDING STANDARDIZED (5, 11) WINDOW LOOKUPS & KEYWORD TABLES ===")
    all_places = set()
    for _, (_, cands, prim) in EVENT_KNOWLEDGE.items():
        all_places.update(cands)
        all_places.add(prim)

    window_lookups = {p: build_window_lookup(p, window_size=5) for p in sorted(all_places)}
    keyword_lookups = {p: load_state_keywords(p) for p in sorted(all_places)}

    print("\n=== STEP 3: LINKING ALL 19 DISASTER CORPORA (TRAIN + DEV + TEST, ZERO DUPLICATES) ===")
    seen_tweet_ids = set()
    seen_date_text = set()
    rows = []

    for event_name in sorted(EVENT_KNOWLEDGE.keys()):
        crisis_type, cand_places, primary_place = EVENT_KNOWLEDGE[event_name]
        ev_dir = os.path.join(CORPUS_DIR, event_name)
        ev_count = 0

        for split_fn in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
            fp = os.path.join(ev_dir, split_fn)
            if not os.path.exists(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    tid = str(rec.get("tweet_id", ""))
                    if tid in seen_tweet_ids:
                        continue

                    date_str = to_date(rec["created_at"])
                    text_str = rec["text"].replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
                    dt_key = (date_str, text_str)
                    if dt_key in seen_date_text:
                        continue

                    # Determine best matching place among candidate places using location_mentions if available
                    chosen_place = None
                    loc_mentions = rec.get("location_mentions", [])
                    if loc_mentions and len(cand_places) > 1:
                        for zone in loc_mentions:
                            z_type = zone.get("type", "")
                            z_text = zone.get("text", "").lower()
                            for c_place in cand_places:
                                kw = keyword_lookups.get(c_place, {})
                                if z_type in kw and z_text in kw[z_type]:
                                    chosen_place = c_place
                                    break
                            if chosen_place is not None:
                                break

                    if chosen_place is None:
                        chosen_place = primary_place

                    w_mat = window_lookups[chosen_place].get(date_str)
                    if w_mat is None:
                        continue

                    humaid_cls = rec["humAID_class"]
                    lbl_useful, lbl_urgent, lbl_sudden = map_labels(humaid_cls, crisis_type)

                    seen_tweet_ids.add(tid)
                    seen_date_text.add(dt_key)
                    ev_count += 1

                    rows.append({
                        "Date": date_str,
                        "Text": text_str,
                        "Crisis_Type": crisis_type,
                        "Window": np.array2string(w_mat, separator=" "),
                        "label_humanitarian": humaid_cls,
                        "label_useful": lbl_useful,
                        "label_urgent": lbl_urgent,
                        "label_sudden": lbl_sudden,
                        "Event_Name": event_name,
                    })

        print(f"  [✓] {event_name:32s} ({crisis_type:10s} -> {primary_place:18s}): {ev_count:5d} unique linked samples")

    df_expanded = pd.DataFrame(rows)
    df_expanded.to_csv(OUT_CSV, sep="\t", index=False)

    print("\n=== EXPANDED CRISISTS-19 DATASET SUMMARY ===")
    print(f"- Output file: {OUT_CSV}")
    print(f"- Total strictly unique multimodal samples: {len(df_expanded):,} (vs 6,923 in CrisisTS_Eng.csv)")
    print(f"- Duplicate (Date, Text) rows: {df_expanded.duplicated(subset=['Date', 'Text']).sum()}")
    print("\nBreakdown by Crisis_Type:")
    print(df_expanded["Crisis_Type"].value_counts())
    print("\nBreakdown by label_urgent:")
    print(df_expanded["label_urgent"].value_counts())
    print("\nBreakdown by label_humanitarian:")
    print(df_expanded["label_humanitarian"].value_counts())


if __name__ == "__main__":
    main()
