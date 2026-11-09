"""
Upgraded Stratified 5-Fold Cross-Validation Benchmark on the Expanded & Cleaned
CrisisTS-19 Dataset (data/CrisisTS_Expanded_19Events.csv, N = 17,628 unique samples across all 19 disasters).

Key Methodological & Performance Upgrades:
1. Cross-Crisis Semantic Prototype & Hazard-Normalized Dual Word+Char TF-IDF (FeatureUnion).
2. 165 Physical, Temporal Volatility, Slope, Acceleration & Extremity Weather Features.
3. Elimination of the Double class_weight='balanced' Meta-Learner Bug in M2_Stacking_Meta_Fusion.
4. In-Fold Out-of-Fold (OOF) Coordinate Threshold / Prior Calibration (Menon et al., ICLR 2021;
   Lipton et al., KDD 2014) strictly fitted on training-fold OOF predictions (Zero Test Leakage).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import re
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import shapiro, ttest_rel, wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_window_3d(w: str) -> np.ndarray:
    vals = [float(x) for x in str(w).replace('[', ' ').replace(']', ' ').split()]
    return np.array(vals, dtype=np.float64).reshape(5, 11)


def extract_rich_ts_features(df_in: pd.DataFrame) -> np.ndarray:
    windows = np.stack([parse_window_3d(w) for w in df_in['Window'].values])  # (N, 5, 11)
    N = len(windows)
    flat = windows.reshape(N, -1)                                          # 55
    mean_c = np.mean(windows, axis=1)                                      # 11
    std_c = np.std(windows, axis=1)                                        # 11
    max_c = np.max(windows, axis=1)                                        # 11
    min_c = np.min(windows, axis=1)                                        # 11
    range_c = max_c - min_c                                                # 11
    slope_c = windows[:, -1, :] - windows[:, 0, :]                         # 11
    diff1 = np.diff(windows, axis=1)                                       # (N, 4, 11)
    accel_c = np.mean(np.diff(diff1, axis=1), axis=1)                      # 11
    abs_mean_c = np.mean(np.abs(windows), axis=1)                          # 11
    abs_max_c = np.max(np.abs(windows), axis=1)                            # 11
    l2_daily = np.linalg.norm(windows, axis=2)                             # 5
    linf_daily = np.max(np.abs(windows), axis=2)                           # 5
    l2_total = np.linalg.norm(flat, axis=1, keepdims=True)                 # 1

    return np.hstack([
        flat, mean_c, std_c, max_c, min_c, range_c,
        slope_c, accel_c, abs_mean_c, abs_max_c,
        l2_daily, linf_daily, l2_total
    ])


def clean_and_enrich_text(text: str) -> str:
    t = str(text)
    t = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', t)
    t = re.sub(r'@\w+', ' [USER] ', t)
    low = t.lower()

    tags = []
    if re.search(r'\b(dead|death|died|killed|killing|fatalit|bodies|corpse|toll|injured|injuries|wounded|victim|casualt)', low):
        tags.append('[SEM_CASUALTY]')
    if re.search(r'\b(destroy|damage|collapsed|collapse|washed away|flooded|burned|burning|ruined|outage|power|bridge|road|house|home|building|roof)', low):
        tags.append('[SEM_DAMAGE]')
    if re.search(r'\b(evacuat|displaced|shelter|flee|stranded|trapped|rescued)', low):
        tags.append('[SEM_EVAC]')
    if re.search(r'\b(warn|warning|alert|advisory|caution|prepare|stay safe|watch|emergency|danger)', low):
        tags.append('[SEM_CAUTION]')
    if re.search(r'\b(missing|found|unaccounted|search and rescue|looking for)', low):
        tags.append('[SEM_MISSING]')
    if re.search(r'\b(donat|volunteer|relief|fund|raise|raising|charity|red cross|unicef|help|support|aid|supply|supplies|food|water|clothes|blood)', low):
        tags.append('[SEM_RELIEF_DONATE]')
    if re.search(r'\b(need|needs|urgent|sos|please help|require|shortage)', low):
        tags.append('[SEM_URGENT_NEED]')

    t = re.sub(r'\s+', ' ', t).strip()
    return f"{t} {' '.join(tags)}"


def optimize_oof_multipliers(oof_proba: np.ndarray, y_true: np.ndarray, n_rounds: int = 2) -> np.ndarray:
    """
    Coordinate ascent on per-class probability multipliers strictly on Out-of-Fold (OOF)
    training predictions to maximize Macro-F1 without test leakage.
    """
    n_classes = oof_proba.shape[1]
    mult = np.ones(n_classes, dtype=np.float64)
    best_score = f1_score(y_true, np.argmax(oof_proba, axis=1), average='macro', zero_division=0)
    grid = np.linspace(0.40, 2.20, 37)

    for _ in range(n_rounds):
        for c in range(n_classes):
            best_c = mult[c]
            for cand in grid:
                mult[c] = cand
                preds = np.argmax(oof_proba * mult[None, :], axis=1)
                sc = f1_score(y_true, preds, average='macro', zero_division=0)
                if sc > best_score + 1e-6:
                    best_score = sc
                    best_c = cand
            mult[c] = best_c
    return mult


def compute_stats(a, b):
    diff = a - b
    _, shapiro_p = shapiro(diff)
    t_stat, t_p = ttest_rel(a, b)
    try:
        _, w_p = wilcoxon(a, b)
    except ValueError:
        w_p = 1.0
    cohen_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
    return {
        'delta_pct': float(np.mean(diff) * 100.0),
        'shapiro_p': float(shapiro_p),
        't_stat': float(t_stat),
        't_p': float(t_p),
        'wilcoxon_p': float(w_p),
        'cohens_d': cohen_d
    }


def main():
    t0 = time.time()
    df = pd.read_csv('data/CrisisTS_Expanded_19Events.csv', sep='\t')
    print(f"Loaded Expanded CrisisTS-19 Dataset: N = {len(df):,} across {df['Event_Name'].nunique()} events", flush=True)
    target_cols = ['label_urgent', 'label_humanitarian', 'label_useful', 'label_sudden']

    X_ts_all = extract_rich_ts_features(df)
    texts_all = df['Text'].apply(clean_and_enrich_text).values

    # Preserve existing loeo_cv_expanded if present
    existing_loeo = None
    try:
        with open('crisists/expanded_benchmark_results.json', 'r', encoding='utf-8') as f:
            existing_loeo = json.load(f).get('loeo_cv_expanded')
    except Exception:
        pass

    results = {
        'dataset_info': {
            'n_samples': int(len(df)),
            'n_events': int(df['Event_Name'].nunique()),
            'crisis_type_counts': {k: int(v) for k, v in df['Crisis_Type'].value_counts().items()},
            'event_counts': {k: int(v) for k, v in df['Event_Name'].value_counts().items()}
        }
    }

    model_names = [
        'B0_Dummy_Majority', 'B1_Dummy_Stratified', 'B2_TS_HistGB', 'B3_TS_RandomForest',
        'B4_Text_LinearSVC', 'B5_Text_LogReg', 'M1_Early_Feature_Fusion', 'M2_Stacking_Meta_Fusion'
    ]

    for target in target_cols:
        t_task = time.time()
        print(f"\n[*] Running Upgraded Stratified 5-Fold CV on Expanded CrisisTS-19 (N={len(df):,}) for {target}...", flush=True)
        results[target] = {}
        y = pd.factorize(df[target], sort=True)[0]
        n_classes = int(np.max(y) + 1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ts_all, y)):
            t_f = time.time()
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_ts_tr = scaler.fit_transform(X_ts_all[train_idx])
            X_ts_te = scaler.transform(X_ts_all[test_idx])

            vec = FeatureUnion([
                ('word', TfidfVectorizer(max_features=22000, ngram_range=(1, 2), sublinear_tf=True)),
                ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=22000, sublinear_tf=True))
            ])
            X_txt_tr = vec.fit_transform(texts_all[train_idx])
            X_txt_te = vec.transform(texts_all[test_idx])

            fold_results = {}

            # B0: Dummy Majority
            b0 = DummyClassifier(strategy='most_frequent')
            b0.fit(X_ts_tr, y_tr)
            p0 = b0.predict(X_ts_te)
            fold_results['B0_Dummy_Majority'] = {
                'macro_f1': f1_score(y_te, p0, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, p0, average='weighted', zero_division=0),
                'accuracy': accuracy_score(y_te, p0)
            }

            # B1: Dummy Stratified
            b1 = DummyClassifier(strategy='stratified', random_state=42)
            b1.fit(X_ts_tr, y_tr)
            p1 = b1.predict(X_ts_te)
            fold_results['B1_Dummy_Stratified'] = {
                'macro_f1': f1_score(y_te, p1, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, p1, average='weighted', zero_division=0),
                'accuracy': accuracy_score(y_te, p1)
            }

            # B2: TS HistGradientBoosting
            b2 = HistGradientBoostingClassifier(max_iter=100, random_state=42)
            b2.fit(X_ts_tr, y_tr)
            p2 = b2.predict(X_ts_te)
            proba_ts_gb_te = b2.predict_proba(X_ts_te)
            fold_results['B2_TS_HistGB'] = {
                'macro_f1': f1_score(y_te, p2, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, p2, average='weighted', zero_division=0),
                'accuracy': accuracy_score(y_te, p2)
            }

            # B3: TS RandomForest
            b3 = RandomForestClassifier(n_estimators=150, max_depth=22, class_weight='balanced', random_state=42, n_jobs=4)
            b3.fit(X_ts_tr, y_tr)
            p3 = b3.predict(X_ts_te)
            proba_ts_rf_te = b3.predict_proba(X_ts_te)
            fold_results['B3_TS_RandomForest'] = {
                'macro_f1': f1_score(y_te, p3, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, p3, average='weighted', zero_division=0),
                'accuracy': accuracy_score(y_te, p3)
            }

            # B4: Text LinearSVC (Calibrated)
            b4 = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', random_state=42, max_iter=1500), cv=3)
            b4.fit(X_txt_tr, y_tr)
            proba_txt_svc_te = b4.predict_proba(X_txt_te)
            p4 = np.argmax(proba_txt_svc_te, axis=1)
            fold_results['B4_Text_LinearSVC'] = {
                'macro_f1': f1_score(y_te, p4, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, p4, average='weighted', zero_division=0),
                'accuracy': accuracy_score(y_te, p4)
            }

            # B5: Text LogisticRegression
            b5 = LogisticRegression(C=2.0, class_weight='balanced', max_iter=500, random_state=42)
            b5.fit(X_txt_tr, y_tr)
            proba_txt_lr_te = b5.predict_proba(X_txt_te)
            p5 = np.argmax(proba_txt_lr_te, axis=1)
            fold_results['B5_Text_LogReg'] = {
                'macro_f1': f1_score(y_te, p5, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, p5, average='weighted', zero_division=0),
                'accuracy': accuracy_score(y_te, p5)
            }

            # M1: Multimodal Early Feature Fusion + In-Fold OOF Threshold Calibration
            X_tr_m1 = hstack([X_txt_tr, csr_matrix(X_ts_tr * 0.12)])
            X_te_m1 = hstack([X_txt_te, csr_matrix(X_ts_te * 0.12)])
            m1_lr = LogisticRegression(C=2.0, class_weight='balanced', max_iter=500, random_state=42)
            m1_svc = CalibratedClassifierCV(LinearSVC(C=0.45, class_weight='balanced', random_state=42, max_iter=1500), cv=3)
            m1_lr.fit(X_tr_m1, y_tr)
            m1_svc.fit(X_tr_m1, y_tr)
            proba_m1_raw_te = 0.50 * m1_lr.predict_proba(X_te_m1) + 0.50 * m1_svc.predict_proba(X_te_m1)

            # Single explicit 3-fold inner OOF loop for both M1 calibration and M2 stacking
            inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            oof_txt = np.zeros((len(y_tr), n_classes), dtype=np.float64)
            oof_m1 = np.zeros((len(y_tr), n_classes), dtype=np.float64)
            oof_ts_rf = np.zeros((len(y_tr), n_classes), dtype=np.float64)
            oof_ts_gb = np.zeros((len(y_tr), n_classes), dtype=np.float64)

            for in_tr, in_va in inner_skf.split(X_ts_tr, y_tr):
                # Text OOF
                c_txt = LogisticRegression(C=2.0, class_weight='balanced', max_iter=400, random_state=42)
                c_txt.fit(X_txt_tr[in_tr], y_tr[in_tr])
                oof_txt[in_va] = c_txt.predict_proba(X_txt_tr[in_va])

                # Early MM OOF (LR + Calibrated LinearSVC)
                c_m1_lr = LogisticRegression(C=2.0, class_weight='balanced', max_iter=400, random_state=42)
                c_m1_svc = CalibratedClassifierCV(LinearSVC(C=0.45, class_weight='balanced', random_state=42, max_iter=1200), cv=2)
                c_m1_lr.fit(X_tr_m1[in_tr], y_tr[in_tr])
                c_m1_svc.fit(X_tr_m1[in_tr], y_tr[in_tr])
                oof_m1[in_va] = 0.50 * c_m1_lr.predict_proba(X_tr_m1[in_va]) + 0.50 * c_m1_svc.predict_proba(X_tr_m1[in_va])

                # TS RF OOF
                c_rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=4)
                c_rf.fit(X_ts_tr[in_tr], y_tr[in_tr])
                oof_ts_rf[in_va] = c_rf.predict_proba(X_ts_tr[in_va])

                # TS HistGB OOF
                c_gb = HistGradientBoostingClassifier(max_iter=80, random_state=42)
                c_gb.fit(X_ts_tr[in_tr], y_tr[in_tr])
                oof_ts_gb[in_va] = c_gb.predict_proba(X_ts_tr[in_va])

            mult_m1 = optimize_oof_multipliers(oof_m1, y_tr)
            pred_m1 = np.argmax(proba_m1_raw_te * mult_m1[None, :], axis=1)
            fold_results['M1_Early_Feature_Fusion'] = {
                'macro_f1': f1_score(y_te, pred_m1, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, pred_m1, average='weighted', zero_division=0),
                'accuracy': accuracy_score(y_te, pred_m1)
            }

            # M2: Upgraded Calibrated OOF Stacking (No double class_weight='balanced' distortion + OOF threshold calibration)
            proba_txt_blend_te = 0.50 * proba_txt_lr_te + 0.50 * proba_txt_svc_te
            S_tr = np.hstack([
                oof_txt, oof_m1, oof_ts_rf, oof_ts_gb,
                oof_txt * oof_ts_rf, oof_m1 * oof_ts_rf
            ])
            S_te = np.hstack([
                proba_txt_blend_te, proba_m1_raw_te, proba_ts_rf_te, proba_ts_gb_te,
                proba_txt_blend_te * proba_ts_rf_te, proba_m1_raw_te * proba_ts_rf_te
            ])

            # Fit meta_clf WITHOUT class_weight='balanced' to avoid squaring minority priors, then calibrate multipliers on OOF
            meta_clf = LogisticRegression(C=1.5, class_weight=None, max_iter=500, random_state=42)
            meta_clf.fit(S_tr, y_tr)
            oof_meta_tr = 0.65 * meta_clf.predict_proba(S_tr) + 0.35 * oof_m1
            proba_meta_te = 0.65 * meta_clf.predict_proba(S_te) + 0.35 * proba_m1_raw_te

            mult_m2 = optimize_oof_multipliers(oof_meta_tr, y_tr)
            pred_m2 = np.argmax(proba_meta_te * mult_m2[None, :], axis=1)
            fold_results['M2_Stacking_Meta_Fusion'] = {
                'macro_f1': f1_score(y_te, pred_m2, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, pred_m2, average='weighted', zero_division=0),
                'accuracy': accuracy_score(y_te, pred_m2)
            }

            results[target][f'fold_{fold_idx}'] = fold_results
            print(
                f"    Fold {fold_idx+1}/5 ({time.time()-t_f:.1f}s) | "
                f"Text={fold_results['B5_Text_LogReg']['macro_f1']:.4f} | "
                f"TS={fold_results['B3_TS_RandomForest']['macro_f1']:.4f} | "
                f"M1={fold_results['M1_Early_Feature_Fusion']['macro_f1']:.4f} ({fold_results['M1_Early_Feature_Fusion']['accuracy']*100:.2f}% Acc) | "
                f"M2={fold_results['M2_Stacking_Meta_Fusion']['macro_f1']:.4f} ({fold_results['M2_Stacking_Meta_Fusion']['accuracy']*100:.2f}% Acc)",
                flush=True
            )

        results[target]['summary'] = {}
        for model in model_names:
            macro_f1s = [results[target][f'fold_{i}'][model]['macro_f1'] for i in range(5)]
            weighted_f1s = [results[target][f'fold_{i}'][model]['weighted_f1'] for i in range(5)]
            accuracies = [results[target][f'fold_{i}'][model]['accuracy'] for i in range(5)]
            results[target]['summary'][model] = {
                'macro_f1_mean': float(np.mean(macro_f1s)),
                'macro_f1_std': float(np.std(macro_f1s, ddof=1)),
                'weighted_f1_mean': float(np.mean(weighted_f1s)),
                'weighted_f1_std': float(np.std(weighted_f1s, ddof=1)),
                'accuracy_mean': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies, ddof=1)),
                'fold_macro_f1s': [float(x) for x in macro_f1s]
            }

        b5_f1s = np.array(results[target]['summary']['B5_Text_LogReg']['fold_macro_f1s'])
        m1_f1s = np.array(results[target]['summary']['M1_Early_Feature_Fusion']['fold_macro_f1s'])
        m2_f1s = np.array(results[target]['summary']['M2_Stacking_Meta_Fusion']['fold_macro_f1s'])

        results[target]['stats'] = {
            'M1_vs_B5': compute_stats(m1_f1s, b5_f1s),
            'M2_vs_B5': compute_stats(m2_f1s, b5_f1s)
        }

        s_b5 = results[target]['summary']['B5_Text_LogReg']
        s_b3 = results[target]['summary']['B3_TS_RandomForest']
        s_m1 = results[target]['summary']['M1_Early_Feature_Fusion']
        s_m2 = results[target]['summary']['M2_Stacking_Meta_Fusion']
        print(f"  [✓] {target} Summary ({time.time()-t_task:.1f}s):", flush=True)
        print(f"      Text_LogReg    : {s_b5['macro_f1_mean']:.4f} ± {s_b5['macro_f1_std']:.4f} (Acc: {s_b5['accuracy_mean']*100:.2f}%)", flush=True)
        print(f"      TS_RandomForest: {s_b3['macro_f1_mean']:.4f} ± {s_b3['macro_f1_std']:.4f} (Acc: {s_b3['accuracy_mean']*100:.2f}%)", flush=True)
        print(f"      M1_Early_Fusion: {s_m1['macro_f1_mean']:.4f} ± {s_m1['macro_f1_std']:.4f} (Acc: {s_m1['accuracy_mean']*100:.2f}%)", flush=True)
        print(f"      M2_Stacking    : {s_m2['macro_f1_mean']:.4f} ± {s_m2['macro_f1_std']:.4f} (Acc: {s_m2['accuracy_mean']*100:.2f}%)", flush=True)

    if existing_loeo is not None:
        results['loeo_cv_expanded'] = existing_loeo

    with open('crisists/expanded_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n[✓] Complete Upgraded Expanded Benchmark saved in {time.time()-t0:.1f}s to crisists/expanded_benchmark_results.json", flush=True)


if __name__ == '__main__':
    main()
