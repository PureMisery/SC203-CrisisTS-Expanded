"""
Multi-Dataset Ingestion and Unified Schema Alignment Pipeline (CrisisTS Extended Ingestion).

Ingests extended disaster tweet corpora:
1. HumAID all 19 events (N = 76,481 baseline from HuggingFace QCRI/HumAID-events).
2. CrisisMMD text & metadata across 7 disaster events (HuggingFace QCRI/CrisisMMD), cleanly omitting image processing.
3. CrisisLex T6 (6 events, N = 49,806) and CrisisLex T26 (26 events, N = 19,893) from HuggingFace QCRI/CrisisBench-all-lang.
4. CrisisNLP curated disaster collections (19 events, N = 40,074) from HuggingFace QCRI/CrisisBench-all-lang.

Key Operations:
- Decodes millisecond UTC dates using 64-bit Twitter Snowflake arithmetic:
    timestamp_ms = (int(tweet_id) >> 22) + 1288834974657
- Harmonizes all diverse native categories into a canonical 10-class humanitarian schema,
  3-class urgency schema, 2-class utility schema, and 3-class suddenness regime.
- Enforces strict global deduplication by `tweet_id` and `(Date, clean_text)`.
- Exports aligned corpora to `data/intermediate_extended_tweets.parquet` and `data/intermediate_extended_tweets.csv.gz`.
"""

import os
import sys
import re
import datetime
import time
import argparse
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

# Default Output Paths
OUT_PARQUET = "data/intermediate_extended_tweets.parquet"
OUT_CSV_GZ = "data/intermediate_extended_tweets.csv.gz"

# Canonical 10-class Humanitarian Schema
CANONICAL_HUMANITARIAN_CLASSES = [
    "rescue_volunteering_or_donation_effort",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "displaced_people_and_evacuations",
    "requests_or_urgent_needs",
    "caution_and_advice",
    "sympathy_and_support",
    "other_relevant_information",
    "not_humanitarian",
]

# Mapping of native text categories into the 10 canonical humanitarian classes
LABEL_HUMANITARIAN_MAP = {
    # HumAID standard 10 classes
    "rescue_volunteering_or_donation_effort": "rescue_volunteering_or_donation_effort",
    "infrastructure_and_utility_damage": "infrastructure_and_utility_damage",
    "injured_or_dead_people": "injured_or_dead_people",
    "missing_or_found_people": "missing_or_found_people",
    "displaced_people_and_evacuations": "displaced_people_and_evacuations",
    "requests_or_urgent_needs": "requests_or_urgent_needs",
    "caution_and_advice": "caution_and_advice",
    "sympathy_and_support": "sympathy_and_support",
    "other_relevant_information": "other_relevant_information",
    "not_humanitarian": "not_humanitarian",
    # CrisisBench / CrisisNLP / CrisisLex variants
    "donation_and_volunteering": "rescue_volunteering_or_donation_effort",
    "response_efforts": "rescue_volunteering_or_donation_effort",
    "infrastructure_and_utilities_damage": "infrastructure_and_utility_damage",
    "physical_landslide": "infrastructure_and_utility_damage",
    "affected_individual": "displaced_people_and_evacuations",
    "displaced_and_evacuations": "displaced_people_and_evacuations",
    "personal_update": "displaced_people_and_evacuations",
    "missing_and_found_people": "missing_or_found_people",
    "requests_or_needs": "requests_or_urgent_needs",
    "disease_related": "other_relevant_information",
    "terrorism_related": "other_relevant_information",
}

# CrisisMMD integer label mapping
CRISISMMD_INT_LABEL_MAP = {
    0: "displaced_people_and_evacuations",       # affected_individuals
    1: "infrastructure_and_utility_damage",     # infrastructure_and_utility_damage
    2: "injured_or_dead_people",                 # injured_or_dead_people
    3: "missing_or_found_people",                # missing_or_found_people
    4: "not_humanitarian",                       # not_humanitarian
    5: "other_relevant_information",             # other_relevant_information
    6: "rescue_volunteering_or_donation_effort", # rescue_volunteering_or_donation_effort
    7: "infrastructure_and_utility_damage",     # vehicle_damage
}

# Event registry: maps native raw event string to (canonical_event_name, Crisis_Type, Hazard_Domain)
# Ensures scientific validity: human-made incidents, transport crashes, and epidemics are
# no longer falsely mapped to natural hazard categories (Earthquake, Wildfire, Flood, Hurricane).
EVENT_METADATA = {
    # HumAID (19 events - All Natural Hazards)
    "california_wildfires_2018": ("california_wildfires_2018", "Wildfire", "Natural_Hazard"),
    "canada_wildfires_2016": ("canada_wildfires_2016", "Wildfire", "Natural_Hazard"),
    "cyclone_idai_2019": ("cyclone_idai_2019", "Hurricane", "Natural_Hazard"),
    "ecuador_earthquake_2016": ("ecuador_earthquake_2016", "Earthquake", "Natural_Hazard"),
    "greece_wildfires_2018": ("greece_wildfires_2018", "Wildfire", "Natural_Hazard"),
    "hurricane_dorian_2019": ("hurricane_dorian_2019", "Hurricane", "Natural_Hazard"),
    "hurricane_florence_2018": ("hurricane_florence_2018", "Hurricane", "Natural_Hazard"),
    "hurricane_harvey_2017": ("hurricane_harvey_2017", "Hurricane", "Natural_Hazard"),
    "hurricane_irma_2017": ("hurricane_irma_2017", "Hurricane", "Natural_Hazard"),
    "hurricane_maria_2017": ("hurricane_maria_2017", "Hurricane", "Natural_Hazard"),
    "hurricane_matthew_2016": ("hurricane_matthew_2016", "Hurricane", "Natural_Hazard"),
    "italy_earthquake_aug_2016": ("italy_earthquake_aug_2016", "Earthquake", "Natural_Hazard"),
    "kaikoura_earthquake_2016": ("kaikoura_earthquake_2016", "Earthquake", "Natural_Hazard"),
    "kerala_floods_2018": ("kerala_floods_2018", "Flood", "Natural_Hazard"),
    "maryland_floods_2018": ("maryland_floods_2018", "Flood", "Natural_Hazard"),
    "midwestern_us_floods_2019": ("midwestern_us_floods_2019", "Flood", "Natural_Hazard"),
    "pakistan_earthquake_2019": ("pakistan_earthquake_2019", "Earthquake", "Natural_Hazard"),
    "puebla_mexico_earthquake_2017": ("puebla_mexico_earthquake_2017", "Earthquake", "Natural_Hazard"),
    "srilanka_floods_2017": ("srilanka_floods_2017", "Flood", "Natural_Hazard"),

    # CrisisMMD (7 events - All Natural Hazards)
    "california_wildfires": ("california_wildfires_2017", "Wildfire", "Natural_Hazard"),
    "hurricane_harvey": ("hurricane_harvey_2017", "Hurricane", "Natural_Hazard"),
    "hurricane_irma": ("hurricane_irma_2017", "Hurricane", "Natural_Hazard"),
    "hurricane_maria": ("hurricane_maria_2017", "Hurricane", "Natural_Hazard"),
    "iraq_iran_earthquake": ("iraq_iran_earthquake_2017", "Earthquake", "Natural_Hazard"),
    "mexico_earthquake": ("puebla_mexico_earthquake_2017", "Earthquake", "Natural_Hazard"),
    "srilanka_floods": ("srilanka_floods_2017", "Flood", "Natural_Hazard"),

    # CrisisLex T6 (6 events)
    "2012_sandy_hurricane-ontopic": ("2012_sandy_hurricane", "Hurricane", "Natural_Hazard"),
    "2012_sandy_hurricane": ("2012_sandy_hurricane", "Hurricane", "Natural_Hazard"),
    "2013_alberta_floods-ontopic": ("2013_alberta_floods", "Flood", "Natural_Hazard"),
    "2013_alberta_floods": ("2013_alberta_floods", "Flood", "Natural_Hazard"),
    "2013_boston_bombings-ontopic": ("2013_boston_bombings", "Terrorism", "Human_Induced"),
    "2013_boston_bombings": ("2013_boston_bombings", "Terrorism", "Human_Induced"),
    "2013_oklahoma_tornado-ontopic": ("2013_oklahoma_tornado", "Tornado", "Natural_Hazard"),
    "2013_oklahoma_tornado": ("2013_oklahoma_tornado", "Tornado", "Natural_Hazard"),
    "2013_queensland_floods-ontopic": ("2013_queensland_floods", "Flood", "Natural_Hazard"),
    "2013_queensland_floods": ("2013_queensland_floods", "Flood", "Natural_Hazard"),
    "2013_west_texas": ("2013_west_texas", "Industrial_Explosion", "Human_Induced"),
    "2013_west-texas_explosion": ("2013_west_texas", "Industrial_Explosion", "Human_Induced"),

    # CrisisLex T26 (26 events)
    "2012_colorado_wildfires": ("2012_colorado_wildfires", "Wildfire", "Natural_Hazard"),
    "2012_costa-rica_earthquake": ("2012_costa_rica_earthquake", "Earthquake", "Natural_Hazard"),
    "2012_guatemala_earthquake": ("2012_guatemala_earthquake", "Earthquake", "Natural_Hazard"),
    "2012_italy_earthquakes": ("2012_italy_earthquakes", "Earthquake", "Natural_Hazard"),
    "2012_philipinnes_floods": ("2012_philippines_floods", "Flood", "Natural_Hazard"),
    "2012_philippines_typhoon-pablo": ("2012_philippines_typhoon_pablo", "Hurricane", "Natural_Hazard"),
    "2012_venezuela_refinery-explosion": ("2012_venezuela_refinery_explosion", "Industrial_Explosion", "Human_Induced"),
    "2013_australia_bushfire": ("2013_australia_bushfire", "Wildfire", "Natural_Hazard"),
    "2013_bangladesh_savar-building-collapse": ("2013_bangladesh_savar_building_collapse", "Structural_Collapse", "Human_Induced"),
    "2013_bohol_earthquake": ("2013_bohol_earthquake", "Earthquake", "Natural_Hazard"),
    "2013_brazil_nightclub-fire": ("2013_brazil_nightclub_fire", "Building_Fire", "Human_Induced"),
    "2013_canada_lac-megantic-train-crash": ("2013_canada_lac_megantic_train_crash", "Transportation_Crash", "Human_Induced"),
    "2013_colorado_floods": ("2013_colorado_floods", "Flood", "Natural_Hazard"),
    "2013_glasgow_helicopter-crash": ("2013_glasgow_helicopter_crash", "Transportation_Crash", "Human_Induced"),
    "2013_italy_sardinia": ("2013_italy_sardinia_floods", "Flood", "Natural_Hazard"),
    "2013_la_airport-shootings": ("2013_la_airport_shootings", "Shooting", "Human_Induced"),
    "2013_manila_floods": ("2013_manila_floods", "Flood", "Natural_Hazard"),
    "2013_ny_train-crash": ("2013_ny_train_crash", "Transportation_Crash", "Human_Induced"),
    "2013_phillipines_typhoon-yolanda": ("2013_philippines_typhoon_yolanda", "Hurricane", "Natural_Hazard"),
    "2013_queensland_floods": ("2013_queensland_floods", "Flood", "Natural_Hazard"),
    "2013_russia_meteor_en-mixed": ("2013_russia_meteor", "Meteor_Impact", "Extraterrestrial"),
    "2013_singapore_haze": ("2013_singapore_haze", "Wildfire", "Natural_Hazard"),
    "2013_spain_train-crash_en-mixed": ("2013_spain_train_crash", "Transportation_Crash", "Human_Induced"),

    # CrisisNLP (19 events across crisisnlp-cf and crisisnlp-volunteers)
    "2013_pakistan_earthquake": ("2013_pakistan_earthquake", "Earthquake", "Natural_Hazard"),
    "2014_california_earthquake": ("2014_california_earthquake", "Earthquake", "Natural_Hazard"),
    "2014_chile_earthquake": ("2014_chile_earthquake", "Earthquake", "Natural_Hazard"),
    "2014_chile_earthquake_esp": ("2014_chile_earthquake", "Earthquake", "Natural_Hazard"),
    "2014_india_floods": ("2014_india_floods", "Flood", "Natural_Hazard"),
    "2014_mexico_hurricane-odile": ("2014_mexico_hurricane_odile", "Hurricane", "Natural_Hazard"),
    "2014_middle-east_respiratory-syndrome": ("2014_middle_east_respiratory_syndrome", "Epidemic", "Health_Emergency"),
    "2014_pakistan_floods": ("2014_pakistan_floods", "Flood", "Natural_Hazard"),
    "2014_philippines_typhoon": ("2014_philippines_typhoon", "Hurricane", "Natural_Hazard"),
    "2014_philippines_typhoon-hagupit": ("2014_philippines_typhoon_hagupit", "Hurricane", "Natural_Hazard"),
    "2014_worldwide_ebola": ("2014_worldwide_ebola", "Epidemic", "Health_Emergency"),
    "2015_nepal_earthquake": ("2015_nepal_earthquake", "Earthquake", "Natural_Hazard"),
    "2015_vanuatu_cyclone": ("2015_vanuatu_cyclone", "Hurricane", "Natural_Hazard"),
    "2015_vanuatu_cyclone-pam": ("2015_vanuatu_cyclone_pam", "Hurricane", "Natural_Hazard"),
    "2014-2015_worldwide_landslides": ("2014_2015_worldwide_landslides", "Landslide", "Natural_Hazard"),
    "2014_iceland_volcano": ("2014_iceland_volcano", "Volcano", "Natural_Hazard"),
    "2014_malaysia_airline": ("2014_malaysia_airline", "Transportation_Crash", "Human_Induced"),
}


def decode_snowflake_date(tid_val) -> str:
    """
    Decodes UTC calendar date string (YYYY-MM-DD) from a 64-bit Twitter Snowflake ID.
    Formula: timestamp_ms = (int(tweet_id) >> 22) + 1288834974657
    """
    try:
        tid = int(tid_val)
        ts_ms = (tid >> 22) + 1288834974657
        dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def clean_text_for_storage(text: str) -> str:
    """Cleans tweet text for tabular TSV / Parquet storage (replaces tabs and newlines)."""
    t = str(text).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", t)


def clean_text_for_dedup(text: str) -> str:
    """Normalizes text for cross-dataset duplicate collision detection."""
    t = clean_text_for_storage(text).lower()
    return " ".join(t.split())


def derive_derived_labels(label_hum: str, crisis_type: str):
    """
    Derives label_useful, label_urgent, and label_sudden from label_humanitarian and crisis_type.
    Strictly conforms to CrisisTS-Full-76k baseline specifications.
    """
    if label_hum == "not_humanitarian":
        return "not_humanitarian", "not_humanitarian", "Not_Crisis_period"
    lbl_useful = "useful"
    lbl_urgent = "not_urgent" if label_hum == "rescue_volunteering_or_donation_effort" else "urgent"
    sudden_types = (
        "Earthquake", "Wildfire", "Terrorism", "Industrial_Explosion",
        "Transportation_Crash", "Structural_Collapse", "Building_Fire",
        "Shooting", "Meteor_Impact", "Volcano", "Tornado"
    )
    lbl_sudden = "Sudden_Crisis" if crisis_type in sudden_types else "Non_Sudden_Crisis"
    return lbl_useful, lbl_urgent, lbl_sudden


def fetch_parquet_cached(url: str, cache_dir: str = "data/cache_ingest", max_retries: int = 5) -> pd.DataFrame:
    """
    Fetches parquet dataframe from local cache or downloads via requests with retries.
    Prevents transient socket aborts (WinError 10053) and minimizes external network traffic.
    """
    os.makedirs(cache_dir, exist_ok=True)
    if "/datasets/" in url:
        safe_name = url.split("/datasets/")[-1].replace("/", "_")
    else:
        safe_name = "_".join([p for p in url.replace("https://", "").split("/") if p])
    local_path = os.path.join(cache_dir, safe_name)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return pd.read_parquet(local_path)

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=45)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                return pd.read_parquet(local_path)
            elif r.status_code == 429:
                time.sleep(2 * attempt)
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts: {e}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: HTTP {r.status_code}")


def ingest_all_datasets(output_parquet=OUT_PARQUET, output_csv_gz=OUT_CSV_GZ) -> pd.DataFrame:
    """
    Executes the end-to-end multi-dataset ingestion, Snowflake date decoding,
    label harmonization, and global deduplication pipeline.
    """
    t0 = time.time()
    seen_tweet_ids = set()
    seen_date_text = set()
    rows = []

    stats = {
        "raw_counts": {},
        "dedup_dropped_id": 0,
        "dedup_dropped_dt": 0,
        "invalid_date": 0,
        "empty_text": 0
    }

    print("=" * 80)
    print("EXTENDED DISASTER TWEET DATASET INGESTION & SCHEMA ALIGNMENT PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Ingest HumAID Baseline (19 events, 76,484 raw rows -> 76,481 aligned)
    # -------------------------------------------------------------------------
    print("\n[1/3] Ingesting HumAID benchmark (19 events) from QCRI/HumAID-events...")
    r = requests.get("https://huggingface.co/api/datasets/QCRI/HumAID-events/parquet", timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch HumAID parquet metadata: HTTP {r.status_code}")
    humaid_meta = r.json()

    humaid_raw = 0
    humaid_aligned = 0
    for event_name in sorted(humaid_meta.keys()):
        canon_event, crisis_type = EVENT_METADATA[event_name]
        splits = humaid_meta[event_name]
        for split_name in ["train", "dev", "test"]:
            urls = splits.get(split_name, [])
            for url in urls:
                df = fetch_parquet_cached(url)
                humaid_raw += len(df)
                for _, rec in df.iterrows():
                    tid = str(rec["tweet_id"])
                    raw_text = clean_text_for_storage(rec.get("tweet_text", ""))
                    if not raw_text:
                        stats["empty_text"] += 1
                        continue
                    if tid in seen_tweet_ids:
                        stats["dedup_dropped_id"] += 1
                        continue
                    date_str = decode_snowflake_date(tid)
                    if not date_str:
                        stats["invalid_date"] += 1
                        continue
                    dt_key = (date_str, clean_text_for_dedup(raw_text))
                    if dt_key in seen_date_text:
                        stats["dedup_dropped_dt"] += 1
                        continue

                    seen_tweet_ids.add(tid)
                    seen_date_text.add(dt_key)

                    label_hum = LABEL_HUMANITARIAN_MAP[str(rec["class_label"])]
                    lbl_useful, lbl_urgent, lbl_sudden = derive_derived_labels(label_hum, crisis_type)

                    rows.append({
                        "tweet_id": tid,
                        "Text": raw_text,
                        "Date": date_str,
                        "Event_Name": canon_event,
                        "Crisis_Type": crisis_type,
                        "Source_Dataset": "HumAID",
                        "label_humanitarian": label_hum,
                        "label_urgent": lbl_urgent,
                        "label_useful": lbl_useful,
                        "label_sudden": lbl_sudden,
                    })
                    humaid_aligned += 1

    stats["raw_counts"]["HumAID"] = humaid_raw
    print(f"  -> HumAID Raw Samples: {humaid_raw:,} | Aligned Unique: {humaid_aligned:,}")

    # -------------------------------------------------------------------------
    # 2. Ingest CrisisMMD (7 events, Text & Metadata only, omitting images)
    # -------------------------------------------------------------------------
    print("\n[2/3] Ingesting CrisisMMD (7 events, Text & Metadata) from QCRI/CrisisMMD...")
    cmmd_raw = 0
    cmmd_aligned = 0
    for split_name in ["train", "dev", "test"]:
        url = f"https://huggingface.co/api/datasets/QCRI/CrisisMMD/parquet/humanitarian/{split_name}/0.parquet"
        df = fetch_parquet_cached(url)
        cmmd_raw += len(df)
        for _, rec in df.iterrows():
            tid = str(rec["tweet_id"])
            raw_text = clean_text_for_storage(rec.get("tweet_text", ""))
            if not raw_text:
                stats["empty_text"] += 1
                continue
            if tid in seen_tweet_ids:
                stats["dedup_dropped_id"] += 1
                continue
            date_str = decode_snowflake_date(tid)
            if not date_str:
                stats["invalid_date"] += 1
                continue
            dt_key = (date_str, clean_text_for_dedup(raw_text))
            if dt_key in seen_date_text:
                stats["dedup_dropped_dt"] += 1
                continue

            ev_raw = str(rec["event_name"])
            canon_event, crisis_type = EVENT_METADATA[ev_raw]
            seen_tweet_ids.add(tid)
            seen_date_text.add(dt_key)

            label_hum = CRISISMMD_INT_LABEL_MAP[int(rec["label"])]
            lbl_useful, lbl_urgent, lbl_sudden = derive_derived_labels(label_hum, crisis_type)

            rows.append({
                "tweet_id": tid,
                "Text": raw_text,
                "Date": date_str,
                "Event_Name": canon_event,
                "Crisis_Type": crisis_type,
                "Source_Dataset": "CrisisMMD",
                "label_humanitarian": label_hum,
                "label_urgent": lbl_urgent,
                "label_useful": lbl_useful,
                "label_sudden": lbl_sudden,
            })
            cmmd_aligned += 1

    stats["raw_counts"]["CrisisMMD"] = cmmd_raw
    print(f"  -> CrisisMMD Raw Samples: {cmmd_raw:,} | Aligned Unique: {cmmd_aligned:,}")

    # -------------------------------------------------------------------------
    # 3. Ingest CrisisBench (CrisisLex T6, CrisisLex T26, CrisisNLP)
    # -------------------------------------------------------------------------
    print("\n[3/3] Ingesting CrisisBench (CrisisLex T6, T26, CrisisNLP) from QCRI/CrisisBench-all-lang...")
    cb_sources = {
        "crisislext6": "CrisisLexT6",
        "crisislext26": "CrisisLexT26",
        "crisisnlp-cf": "CrisisNLP",
        "crisisnlp-volunteers": "CrisisNLP",
    }
    cb_raw = {s: 0 for s in set(cb_sources.values())}
    cb_aligned = {s: 0 for s in set(cb_sources.values())}

    for split_name in ["train", "dev", "test"]:
        url = f"https://huggingface.co/api/datasets/QCRI/CrisisBench-all-lang/parquet/humanitarian/{split_name}/0.parquet"
        df = fetch_parquet_cached(url)
        for _, rec in df.iterrows():
            src = str(rec["source"])
            if src not in cb_sources:
                continue
            src_dataset = cb_sources[src]
            cb_raw[src_dataset] += 1

            tid = str(rec["id"])
            raw_text = clean_text_for_storage(rec.get("text", ""))
            if not raw_text:
                stats["empty_text"] += 1
                continue
            if tid in seen_tweet_ids:
                stats["dedup_dropped_id"] += 1
                continue
            date_str = decode_snowflake_date(tid)
            if not date_str:
                stats["invalid_date"] += 1
                continue
            dt_key = (date_str, clean_text_for_dedup(raw_text))
            if dt_key in seen_date_text:
                stats["dedup_dropped_dt"] += 1
                continue

            ev_raw = str(rec["event"])
            canon_event, crisis_type, hazard_domain = EVENT_METADATA[ev_raw]
            seen_tweet_ids.add(tid)
            seen_date_text.add(dt_key)

            label_hum = LABEL_HUMANITARIAN_MAP[str(rec["class_label"])]
            lbl_useful, lbl_urgent, lbl_sudden = derive_derived_labels(label_hum, crisis_type)

            rows.append({
                "tweet_id": tid,
                "Text": raw_text,
                "Date": date_str,
                "Event_Name": canon_event,
                "Crisis_Type": crisis_type,
                "Hazard_Domain": hazard_domain,
                "Source_Dataset": src_dataset,
                "label_humanitarian": label_hum,
                "label_urgent": lbl_urgent,
                "label_useful": lbl_useful,
                "label_sudden": lbl_sudden,
            })
            cb_aligned[src_dataset] += 1

    for k, v in cb_raw.items():
        stats["raw_counts"][k] = v
        print(f"  -> {k} Raw Samples: {v:,} | Aligned Unique: {cb_aligned[k]:,}")

    df_full = pd.DataFrame(rows)

    # Reorder columns strictly according to interface contract
    cols_order = [
        "tweet_id", "Text", "Date", "Event_Name", "Crisis_Type", "Hazard_Domain",
        "Source_Dataset", "label_humanitarian", "label_urgent",
        "label_useful", "label_sudden"
    ]
    df_full = df_full[cols_order]

    # Verify zero nulls
    null_counts = df_full.isna().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Integrity Error: Found null values in aligned dataset:\n{null_counts}")

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_gz), exist_ok=True)

    print("\n" + "=" * 80)
    print("SERIALIZING HARMONIZED MULTI-DATASET CORPUS")
    print("=" * 80)
    print(f"Writing Parquet: {output_parquet}...")
    df_full.to_parquet(output_parquet, index=False, engine="pyarrow", compression="snappy")
    print(f"  -> Saved {output_parquet} ({os.path.getsize(output_parquet) / (1024 * 1024):.2f} MB)")

    print(f"Writing Compressed CSV: {output_csv_gz}...")
    df_full.to_csv(output_csv_gz, sep="\t", index=False, compression="gzip")
    print(f"  -> Saved {output_csv_gz} ({os.path.getsize(output_csv_gz) / (1024 * 1024):.2f} MB)")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("INGESTION & HARMONIZATION SUMMARY")
    print("=" * 80)
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print(f"Total Aligned Samples: N = {len(df_full):,}")
    print(f"Total Disaster Events: {df_full['Event_Name'].nunique():,} events")
    print(f"Total Duplicate Tweet IDs Filtered: {stats['dedup_dropped_id']:,}")
    print(f"Total Date-Text Duplicate Collisions Filtered: {stats['dedup_dropped_dt']:,}")

    print("\n[Distribution: Source Dataset]")
    for src, count in df_full["Source_Dataset"].value_counts().items():
        print(f"  - {src:16s}: {count:7,} ({count / len(df_full) * 100:5.2f}%)")

    print("\n[Distribution: Hazard Domain]")
    for hd, count in df_full["Hazard_Domain"].value_counts().items():
        print(f"  - {hd:20s}: {count:7,} ({count / len(df_full) * 100:5.2f}%)")

    print("\n[Distribution: Crisis Type (Macro Hazards)]")
    for ct, count in df_full["Crisis_Type"].value_counts().items():
        print(f"  - {ct:24s}: {count:7,} ({count / len(df_full) * 100:5.2f}%)")

    print("\n[Distribution: label_humanitarian (10 Classes)]")
    for lh, count in df_full["label_humanitarian"].value_counts().items():
        print(f"  - {lh:38s}: {count:7,} ({count / len(df_full) * 100:5.2f}%)")

    print("\n[Distribution: label_urgent (3 Classes)]")
    for lu, count in df_full["label_urgent"].value_counts().items():
        print(f"  - {lu:16s}: {count:7,} ({count / len(df_full) * 100:5.2f}%)")

    print("\n[Distribution: label_useful (2 Classes)]")
    for luse, count in df_full["label_useful"].value_counts().items():
        print(f"  - {luse:16s}: {count:7,} ({count / len(df_full) * 100:5.2f}%)")

    print("\n[Distribution: label_sudden (3 Classes)]")
    for ls, count in df_full["label_sudden"].value_counts().items():
        print(f"  - {ls:18s}: {count:7,} ({count / len(df_full) * 100:5.2f}%)")

    return df_full


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Dataset Extended Disaster Tweet Ingestion Pipeline")
    parser.add_argument("--output_parquet", default=OUT_PARQUET, help="Output Parquet path")
    parser.add_argument("--output_csv_gz", default=OUT_CSV_GZ, help="Output compressed CSV path")
    args = parser.parse_args()

    ingest_all_datasets(args.output_parquet, args.output_csv_gz)
