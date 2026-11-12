import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import shapiro, ttest_rel, wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Load data
df = pd.read_csv('data/CrisisTS_Eng.csv', sep='\t')
target_cols = ['label_urgent', 'label_humanitarian', 'label_useful', 'label_sudden']


def parse_window_3d(w: str) -> np.ndarray:
    vals = [float(x) for x in str(w).replace('[', ' ').replace(']', ' ').split()]
    return np.array(vals, dtype=np.float64).reshape(5, 11)


def extract_rich_ts_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts 165 physical & temporal features from the (5, 11) meteorological window:
    - Raw flattened window (55)
    - Channel mean, std, max, min, range, last-minus-first slope, day-to-day acceleration (77)
    - Absolute anomaly magnitude mean & max (22)
    - Daily L2 energy (5), Daily L_inf peak (5), Total L2 energy (1)
    """
    windows = np.stack([parse_window_3d(w) for w in df['Window'].values])  # (N, 5, 11)
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


def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', str(text))
    text = re.sub(r'@\w+', ' [USER] ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


X_ts_all = extract_rich_ts_features(df)
texts_all = df['Text'].apply(clean_text).values

results = {}

for target in target_cols:
    print(f"Running 5-Fold CV for {target}...")
    results[target] = {}
    y = pd.factorize(df[target], sort=True)[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ts_all, y)):
        y_tr, y_te = y[train_idx], y[test_idx]

        # Strictly fit scalers and vectorizers inside the training fold (Zero Leakage)
        scaler = StandardScaler()
        X_ts_tr = scaler.fit_transform(X_ts_all[train_idx])
        X_ts_te = scaler.transform(X_ts_all[test_idx])

        vec = FeatureUnion([
            ('word', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=25000, sublinear_tf=True))
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
        b2 = HistGradientBoostingClassifier(random_state=42)
        b2.fit(X_ts_tr, y_tr)
        p2 = b2.predict(X_ts_te)
        proba_ts_gb_te = b2.predict_proba(X_ts_te)
        fold_results['B2_TS_HistGB'] = {
            'macro_f1': f1_score(y_te, p2, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_te, p2, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_te, p2)
        }

        # B3: TS RandomForest
        b3 = RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=42, n_jobs=-1)
        b3.fit(X_ts_tr, y_tr)
        p3 = b3.predict(X_ts_te)
        proba_ts_rf_te = b3.predict_proba(X_ts_te)
        fold_results['B3_TS_RandomForest'] = {
            'macro_f1': f1_score(y_te, p3, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_te, p3, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_te, p3)
        }

        # B4: Text LinearSVC (Calibrated)
        b4 = CalibratedClassifierCV(LinearSVC(C=0.6, class_weight='balanced', random_state=42, max_iter=2000), cv=3)
        b4.fit(X_txt_tr, y_tr)
        p4 = b4.predict(X_txt_te)
        proba_txt_svc_te = b4.predict_proba(X_txt_te)
        fold_results['B4_Text_LinearSVC'] = {
            'macro_f1': f1_score(y_te, p4, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_te, p4, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_te, p4)
        }

        # B5: Text LogisticRegression
        b5 = LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000, random_state=42)
        b5.fit(X_txt_tr, y_tr)
        p5 = b5.predict(X_txt_te)
        proba_txt_lr_te = b5.predict_proba(X_txt_te)
        fold_results['B5_Text_LogReg'] = {
            'macro_f1': f1_score(y_te, p5, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_te, p5, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_te, p5)
        }

        # M1: Multimodal Early Feature Fusion (Word+Char TF-IDF + 165 Rich TS Features)
        X_tr_m1 = hstack([X_txt_tr, csr_matrix(X_ts_tr * 0.12)])
        X_te_m1 = hstack([X_txt_te, csr_matrix(X_ts_te * 0.12)])
        m1_lr = LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000, random_state=42)
        m1_svc = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', random_state=42, max_iter=2000), cv=3)
        m1_lr.fit(X_tr_m1, y_tr)
        m1_svc.fit(X_tr_m1, y_tr)
        proba_m1_te = 0.55 * m1_lr.predict_proba(X_te_m1) + 0.45 * m1_svc.predict_proba(X_te_m1)
        pred_m1 = np.argmax(proba_m1_te, axis=1)
        fold_results['M1_Early_Feature_Fusion'] = {
            'macro_f1': f1_score(y_te, pred_m1, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_te, pred_m1, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_te, pred_m1)
        }

        # M2: Improved Out-of-Fold Calibrated Stacking Meta-Fusion ([P_txt, P_m1, P_ts_rf, P_ts_gb, P_txt * P_ts_rf])
        oof_txt = cross_val_predict(
            LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000, random_state=42),
            X_txt_tr, y_tr, cv=3, method='predict_proba', n_jobs=-1
        )
        oof_m1 = cross_val_predict(
            LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000, random_state=42),
            X_tr_m1, y_tr, cv=3, method='predict_proba', n_jobs=-1
        )
        oof_ts_rf = cross_val_predict(
            RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=-1),
            X_ts_tr, y_tr, cv=3, method='predict_proba', n_jobs=-1
        )
        oof_ts_gb = cross_val_predict(
            HistGradientBoostingClassifier(random_state=42),
            X_ts_tr, y_tr, cv=3, method='predict_proba', n_jobs=-1
        )

        S_tr = np.hstack([oof_txt, oof_m1, oof_ts_rf, oof_ts_gb, oof_txt * oof_ts_rf])
        S_te = np.hstack([proba_txt_lr_te, proba_m1_te, proba_ts_rf_te, proba_ts_gb_te, proba_txt_lr_te * proba_ts_rf_te])

        meta_clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        meta_clf.fit(S_tr, y_tr)
        pred_m2 = meta_clf.predict(S_te)
        fold_results['M2_Stacking_Meta_Fusion'] = {
            'macro_f1': f1_score(y_te, pred_m2, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_te, pred_m2, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_te, pred_m2)
        }

        results[target][f'fold_{fold_idx}'] = fold_results

model_names = [
    'B0_Dummy_Majority', 'B1_Dummy_Stratified', 'B2_TS_HistGB', 'B3_TS_RandomForest',
    'B4_Text_LinearSVC', 'B5_Text_LogReg', 'M1_Early_Feature_Fusion', 'M2_Stacking_Meta_Fusion'
]

for target in target_cols:
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

    results[target]['stats'] = {
        'M1_vs_B5': compute_stats(m1_f1s, b5_f1s),
        'M2_vs_B5': compute_stats(m2_f1s, b5_f1s)
    }

with open('crisists/full_benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

md_lines = ['# Full CrisisTS Benchmark Results (Upgraded Word+Char TF-IDF + 165 TS Features + OOF Stacking)\n']
for target in target_cols:
    md_lines.append(f'## {target}\n')
    md_lines.append('| Model | Macro-F1 (Mean ± Std) | Weighted-F1 (Mean ± Std) | Accuracy (Mean ± Std) |')
    md_lines.append('| :--- | :---: | :---: | :---: |')
    for model in model_names:
        s = results[target]['summary'][model]
        md_lines.append(
            f"| **{model}** | {s['macro_f1_mean']:.4f} ± {s['macro_f1_std']:.4f} | "
            f"{s['weighted_f1_mean']:.4f} ± {s['weighted_f1_std']:.4f} | "
            f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f} |"
        )
    md_lines.append('\n### Statistical Tests vs. `B5_Text_LogReg`\n')
    md_lines.append('| Comparison | Δ Macro-F1 | Paired t-stat | t-test p-value | Wilcoxon p-value | Cohen\'s d |')
    md_lines.append('| :--- | :---: | :---: | :---: | :---: | :---: |')
    for k, label in [('M1_vs_B5', 'M1_Early_Feature_Fusion vs B5'), ('M2_vs_B5', 'M2_Stacking_Meta_Fusion vs B5')]:
        st = results[target]['stats'][k]
        md_lines.append(
            f"| {label} | {st['delta_pct']:+.2f}% | {st['t_stat']:.4f} | {st['t_p']:.6f} | {st['wilcoxon_p']:.4f} | {st['cohens_d']:.4f} |"
        )
    md_lines.append('\n')

with open('crisists/full_benchmark_tables.md', 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(md_lines))

print('\n'.join(md_lines))
