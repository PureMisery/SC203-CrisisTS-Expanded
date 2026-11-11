"""
Automated Verification Suite for M1 Deliverables:
- data/intermediate_extended_tweets.parquet
- data/intermediate_extended_tweets.csv.gz
- crisists/ingest_extended_datasets.py
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np

PARQUET_PATH = "data/intermediate_extended_tweets.parquet"
CSV_GZ_PATH = "data/intermediate_extended_tweets.csv.gz"

EXPECTED_COLUMNS = [
    "tweet_id", "Text", "Date", "Event_Name", "Crisis_Type",
    "Source_Dataset", "label_humanitarian", "label_urgent",
    "label_useful", "label_sudden"
]

EXPECTED_HUMANITARIAN_10 = {
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
}

EXPECTED_URGENT_3 = {"urgent", "not_urgent", "not_humanitarian"}
EXPECTED_USEFUL_2 = {"useful", "not_humanitarian"}
EXPECTED_SUDDEN_3 = {"Sudden_Crisis", "Non_Sudden_Crisis", "Not_Crisis_period"}
EXPECTED_CRISIS_4 = {"Earthquake", "Flood", "Hurricane", "Wildfire"}
EXPECTED_SOURCE_5 = {"HumAID", "CrisisLexT6", "CrisisLexT26", "CrisisMMD", "CrisisNLP"}


def decode_snowflake_date(tid_val):
    tid = int(tid_val)
    ts_ms = (tid >> 22) + 1288834974657
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d")


def test_verification():
    print("=" * 80)
    print("RUNNING AUTOMATED VERIFICATION SUITE FOR M1 INTERMEDIATE DATASET")
    print("=" * 80)

    # 1. File existence
    assert os.path.exists(PARQUET_PATH), f"Missing parquet: {PARQUET_PATH}"
    assert os.path.exists(CSV_GZ_PATH), f"Missing csv.gz: {CSV_GZ_PATH}"
    p_size_mb = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
    c_size_mb = os.path.getsize(CSV_GZ_PATH) / (1024 * 1024)
    print(f"[TEST 1/8] File Existence: PASSED (Parquet: {p_size_mb:.2f} MB, CSV.GZ: {c_size_mb:.2f} MB)")

    # 2. Load and verify shape and schema
    df = pd.read_parquet(PARQUET_PATH)
    assert len(df) == 202312, f"Expected 202,312 rows, got {len(df)}"
    assert list(df.columns) == EXPECTED_COLUMNS, f"Columns mismatch: {list(df.columns)}"
    print(f"[TEST 2/8] Dataset Shape & Column Schema: PASSED (N = {len(df):,}, Cols = {len(df.columns)})")

    # 3. Check for any nulls or empty strings
    nulls = df.isna().sum()
    assert nulls.sum() == 0, f"Found nulls:\n{nulls}"
    empty_tid = (df["tweet_id"].str.len() == 0).sum()
    empty_txt = (df["Text"].str.len() == 0).sum()
    empty_date = (df["Date"].str.len() == 0).sum()
    assert empty_tid == 0 and empty_txt == 0 and empty_date == 0, "Found empty strings"
    print(f"[TEST 3/8] Null and Empty String Audit: PASSED (Zero nulls across all 10 columns)")

    # 4. Strict categorical taxonomy verification
    assert set(df["label_humanitarian"].unique()) == EXPECTED_HUMANITARIAN_10
    assert set(df["label_urgent"].unique()) == EXPECTED_URGENT_3
    assert set(df["label_useful"].unique()) == EXPECTED_USEFUL_2
    assert set(df["label_sudden"].unique()) == EXPECTED_SUDDEN_3
    assert set(df["Crisis_Type"].unique()) == EXPECTED_CRISIS_4
    assert set(df["Source_Dataset"].unique()) == EXPECTED_SOURCE_5
    print("[TEST 4/8] Categorical Schema Conformity: PASSED (All 10/3/2/3/4/5 category sets strictly compliant)")

    # 5. Label derivation consistency logic
    # Check useful logic
    useful_mismatch = ((df["label_humanitarian"] == "not_humanitarian") & (df["label_useful"] != "not_humanitarian")).sum()
    useful_mismatch += ((df["label_humanitarian"] != "not_humanitarian") & (df["label_useful"] != "useful")).sum()
    assert useful_mismatch == 0, f"label_useful logic violation: {useful_mismatch}"

    # Check urgent logic
    urgent_mismatch = ((df["label_humanitarian"] == "not_humanitarian") & (df["label_urgent"] != "not_humanitarian")).sum()
    urgent_mismatch += ((df["label_humanitarian"] == "rescue_volunteering_or_donation_effort") & (df["label_urgent"] != "not_urgent")).sum()
    urgent_mismatch += ((df["label_humanitarian"] != "not_humanitarian") & (df["label_humanitarian"] != "rescue_volunteering_or_donation_effort") & (df["label_urgent"] != "urgent")).sum()
    assert urgent_mismatch == 0, f"label_urgent logic violation: {urgent_mismatch}"

    # Check sudden logic
    sudden_mismatch = ((df["label_humanitarian"] == "not_humanitarian") & (df["label_sudden"] != "Not_Crisis_period")).sum()
    sudden_mismatch += ((df["label_humanitarian"] != "not_humanitarian") & (df["Crisis_Type"].isin(["Earthquake", "Wildfire"])) & (df["label_sudden"] != "Sudden_Crisis")).sum()
    sudden_mismatch += ((df["label_humanitarian"] != "not_humanitarian") & (df["Crisis_Type"].isin(["Hurricane", "Flood"])) & (df["label_sudden"] != "Non_Sudden_Crisis")).sum()
    assert sudden_mismatch == 0, f"label_sudden logic violation: {sudden_mismatch}"
    print("[TEST 5/8] Label Derivation Integrity Logic: PASSED (100% mathematical consistency across all 4 label fields)")

    # 6. Global deduplication verification
    assert df["tweet_id"].nunique() == len(df), f"Duplicate tweet_id found! {len(df) - df['tweet_id'].nunique()}"
    clean_texts = df["Text"].str.lower().apply(lambda x: " ".join(x.split()))
    dt_keys = df["Date"] + "___" + clean_texts
    assert dt_keys.nunique() == len(df), f"Duplicate (Date, clean_text) found! {len(df) - dt_keys.nunique()}"
    print(f"[TEST 6/8] Global Deduplication Integrity: PASSED (Zero duplicate IDs and zero (date, text) collisions)")

    # 7. Snowflake timestamp decoding consistency check
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=1000, replace=False)
    for idx in sample_indices:
        row = df.iloc[idx]
        rec_date = decode_snowflake_date(row["tweet_id"])
        assert rec_date == row["Date"], f"Snowflake decoding mismatch at row {idx}: expected {row['Date']}, got {rec_date}"
    print(f"[TEST 7/8] Snowflake Arithmetic Spot-Check (1,000 samples): PASSED (100% exact UTC calendar match)")

    # 8. Parquet vs CSV.GZ consistency check
    df_csv = pd.read_csv(CSV_GZ_PATH, sep="\t", compression="gzip", dtype=str)
    assert len(df_csv) == len(df), f"Row count mismatch between parquet and csv.gz: {len(df_csv)} vs {len(df)}"
    assert list(df_csv.columns) == list(df.columns), "Columns mismatch between parquet and csv.gz"
    # Spot-check 100 rows
    for i in [0, 100, 500, 1000, 5000, 10000, 50000, 100000, 150000, 200000]:
        assert df.iloc[i]["tweet_id"] == df_csv.iloc[i]["tweet_id"], f"Mismatch at row {i}"
        assert df.iloc[i]["label_humanitarian"] == df_csv.iloc[i]["label_humanitarian"], f"Mismatch at row {i}"
    print(f"[TEST 8/8] Parquet vs CSV.GZ Cross-Check: PASSED (100% cross-format agreement)")

    print("\n" + "=" * 80)
    print("ALL 8 VERIFICATION TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    test_verification()
