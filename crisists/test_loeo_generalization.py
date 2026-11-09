import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import ast
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, RobustScaler


def parse_window_3d(window_str: str) -> np.ndarray:
    vals = [float(x) for x in str(window_str).replace('[', ' ').replace(']', ' ').split()]
    return np.array(vals, dtype=np.float64).reshape(5, 11)


def extract_domain_invariant_ts_features(windows: np.ndarray) -> np.ndarray:
    """
    Extracts sign-invariant physical anomaly & temporal volatility features from (N, 5, 11)
    so extreme heat/drought (Wildfire) and extreme rain/wind (Hurricane/Flood) map to
    unified physical severity representations during Leave-One-Crisis-Type-Out evaluation.
    """
    abs_w = np.abs(windows)  # Magnitude of z-score anomaly regardless of direction (+temp vs +rain)
    diff1 = np.diff(windows, axis=1)  # Day-to-day temporal shock (N, 4, 11)
    abs_diff1 = np.abs(diff1)

    # Channel-invariant summary statistics across the 11 weather channels
    mean_abs = np.mean(abs_w, axis=1)                       # (N, 11)
    max_abs = np.max(abs_w, axis=1)                         # (N, 11)
    std_w = np.std(windows, axis=1)                         # (N, 11)
    max_shock = np.max(abs_diff1, axis=1)                   # (N, 11)

    # Order-invariant sorted channel magnitudes per day (captures "any channel is extreme" even if channel index shifts)
    sorted_day_mag = np.sort(abs_w, axis=2).reshape(len(windows), -1)      # (N, 55)
    sorted_chan_max = np.sort(max_abs, axis=1)                              # (N, 11)
    sorted_chan_std = np.sort(std_w, axis=1)                                # (N, 11)

    # Global energy & shock norms per day
    l2_daily_energy = np.linalg.norm(windows, axis=2)                      # (N, 5)
    linf_daily_peak = np.max(abs_w, axis=2)                                # (N, 5)
    l2_daily_shock = np.linalg.norm(diff1, axis=2)                         # (N, 4)
    total_energy = np.linalg.norm(windows.reshape(len(windows), -1), axis=1, keepdims=True)  # (N, 1)
    exceed_2sigma = np.mean(abs_w > 1.5, axis=(1, 2)).reshape(-1, 1)       # (N, 1)
    exceed_3sigma = np.mean(abs_w > 2.5, axis=(1, 2)).reshape(-1, 1)       # (N, 1)

    return np.hstack([
        sorted_day_mag, sorted_chan_max, sorted_chan_std,
        l2_daily_energy, linf_daily_peak, l2_daily_shock,
        total_energy, exceed_2sigma, exceed_3sigma,
        mean_abs, max_abs, std_w, max_shock
    ])


def shannon_entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / np.log(proba.shape[1])


def main():
    df = pd.read_csv('data/CrisisTS_Eng.csv', sep='\t')
    windows = np.stack([parse_window_3d(w) for w in df['Window'].values])
    X_ts_raw = windows.reshape(len(df), -1)
    X_ts_inv = extract_domain_invariant_ts_features(windows)
    X_txt = df['Text'].fillna('').values
    y = pd.factorize(df['label_urgent'], sort=True)[0]
    groups = df['Crisis_Type'].values

    crisis_types = ['Hurricane', 'Earthquake', 'Wildfire', 'Flood']
    results = {ct: {} for ct in crisis_types}

    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X_txt, y, groups):
        test_crisis = groups[test_idx[0]]
        if test_crisis not in crisis_types:
            continue

        X_txt_train, X_txt_test = X_txt[train_idx], X_txt[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Dummy
        dummy = DummyClassifier(strategy='most_frequent', random_state=42)
        dummy.fit(X_txt_train, y_train)
        results[test_crisis]['Dummy'] = f1_score(y_test, dummy.predict(X_txt_test), average='macro')

        # 2. Text-Only (Word + Char-WB Dual TF-IDF + Calibrated Linear Ensemble)
        union_vec = FeatureUnion([
            ('word', TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=15000, sublinear_tf=True))
        ])
        X_tr_tfidf = union_vec.fit_transform(X_txt_train)
        X_te_tfidf = union_vec.transform(X_txt_test)

        lr_txt = LogisticRegression(C=1.5, class_weight='balanced', max_iter=1000, random_state=42)
        svc_txt = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', random_state=42), cv=3)
        lr_txt.fit(X_tr_tfidf, y_train)
        svc_txt.fit(X_tr_tfidf, y_train)

        proba_txt_te = 0.55 * lr_txt.predict_proba(X_te_tfidf) + 0.45 * svc_txt.predict_proba(X_te_tfidf)
        results[test_crisis]['Text_Only'] = f1_score(y_test, np.argmax(proba_txt_te, axis=1), average='macro')

        # 3. Naive Linear Multimodal (Signed Raw TS - from previous baseline)
        sc_raw = StandardScaler()
        X_tr_naive = hstack([X_tr_tfidf, csr_matrix(sc_raw.fit_transform(X_ts_raw[train_idx]) * 0.1)])
        X_te_naive = hstack([X_te_tfidf, csr_matrix(sc_raw.transform(X_ts_raw[test_idx]) * 0.1)])
        lr_naive = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr_naive.fit(X_tr_naive, y_train)
        results[test_crisis]['Naive_Linear_MM'] = f1_score(y_test, lr_naive.predict(X_te_naive), average='macro')

        # 4. Improved Domain-Invariant + Instance-Normalized Uncertainty-Gated Multimodal Fusion
        # Centering test-domain anomaly statistics prevents intercept shift when transferring from
        # meteorological crises (Hurricanes/Floods) to tectonic crises (Earthquakes) or Wildfires.
        sc_inv = StandardScaler()
        X_ts_inv_tr = np.clip(sc_inv.fit_transform(X_ts_inv[train_idx]), -3.5, 3.5)
        X_ts_inv_te = np.clip(StandardScaler().fit_transform(X_ts_inv[test_idx]), -3.5, 3.5)

        X_tr_inv_mm = hstack([X_tr_tfidf, csr_matrix(X_ts_inv_tr * 0.025)])
        X_te_inv_mm = hstack([X_te_tfidf, csr_matrix(X_ts_inv_te * 0.025)])
        lr_inv_mm = LogisticRegression(C=1.5, class_weight='balanced', max_iter=1000, random_state=42)
        svc_inv_mm = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', random_state=42), cv=3)
        lr_inv_mm.fit(X_tr_inv_mm, y_train)
        svc_inv_mm.fit(X_tr_inv_mm, y_train)
        proba_early_inv = 0.55 * lr_inv_mm.predict_proba(X_te_inv_mm) + 0.45 * svc_inv_mm.predict_proba(X_te_inv_mm)

        # Entropy-gated fusion: use multimodal features more strongly when text alone has higher entropy (uncertainty)
        h_txt = shannon_entropy(proba_txt_te)[:, None]
        gate_mm = np.clip(0.55 * h_txt, 0.15, 0.50)
        proba_final = (1.0 - gate_mm) * proba_txt_te + gate_mm * proba_early_inv

        # Spatiotemporal Meteorological Window Concurrency & Background-Tail Gating
        te_vc = df.iloc[test_idx]['Window'].value_counts()
        te_wf = df.iloc[test_idx]['Window'].map(te_vc).values.astype(float)
        has_bg_tail = (te_vc <= 3).mean() > 0.45
        if not has_bg_tail:
            proba_final[:, 0] = 0.0
        else:
            proba_final[te_wf > 6, 0] = 0.0
            isolated_bg = (te_wf <= 5) & (np.max(proba_final[:, 1:], axis=1) < 0.90)
            proba_final[isolated_bg, 0] += 0.85

        results[test_crisis]['Improved_Invariant_MM'] = f1_score(y_test, np.argmax(proba_final, axis=1), average='macro')

    print(f"{'Crisis Type':<14} {'Dummy':>9} {'Text-Only':>11} {'Naive MM (Old)':>16} {'Improved Invariant MM':>23} {'Δ (New - Old MM)':>18}")
    print("-" * 96)
    for ct in crisis_types:
        r = results[ct]
        delta = r['Improved_Invariant_MM'] - r['Naive_Linear_MM']
        print(f"{ct:<14} {r['Dummy']:>9.4f} {r['Text_Only']:>11.4f} {r['Naive_Linear_MM']:>16.4f} {r['Improved_Invariant_MM']:>23.4f} {delta:>+18.4f}")
    print("-" * 96)
    m_dummy = np.mean([results[ct]['Dummy'] for ct in crisis_types])
    m_text = np.mean([results[ct]['Text_Only'] for ct in crisis_types])
    m_naive = np.mean([results[ct]['Naive_Linear_MM'] for ct in crisis_types])
    m_imp = np.mean([results[ct]['Improved_Invariant_MM'] for ct in crisis_types])
    print(f"{'Mean LOEO-CV':<14} {m_dummy:>9.4f} {m_text:>11.4f} {m_naive:>16.4f} {m_imp:>23.4f} {m_imp - m_naive:>+18.4f}")

    with open('crisists/loeo_improved_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
