import re
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


def parse_window(s: str) -> np.ndarray:
    """Parse Window string [[...]] of shape (5,11) into numpy array."""
    s = s.strip()
    s = re.sub(r'\s+', ',', s)
    s = s.replace('[,', '[').replace(',]', ']')
    arr = ast.literal_eval(s)
    return np.array(arr, dtype=np.float32)


def extract_ts_features(windows: np.ndarray) -> np.ndarray:
    """
    Extract 110 features from windows of shape (N, 5, 11):
    - flat: 55
    - mean per channel: 11
    - std per channel: 11
    - max per channel: 11
    - min per channel: 11
    - delta_last_minus_first per channel: 11
    Total: 110
    """
    N, T, C = windows.shape
    flat = windows.reshape(N, T * C)  # 55
    mean = windows.mean(axis=1)       # 11
    std = windows.std(axis=1)         # 11
    maxv = windows.max(axis=1)        # 11
    minv = windows.min(axis=1)        # 11
    delta = windows[:, -1, :] - windows[:, 0, :]  # 11
    features = np.hstack([flat, mean, std, maxv, minv, delta])
    return features.astype(np.float32)


def compute_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy H(p) = -sum(p * log(p + eps)) per sample."""
    eps = 1e-12
    return -np.sum(proba * np.log(proba + eps), axis=1)


def main():
    data_path = Path('data/CrisisTS_Eng.csv')
    out_path = Path('crisists/nonlinear_ablation_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path, sep='\t')
    texts = df['Text'].astype(str).values
    y = pd.factorize(df['label_urgent'], sort=True)[0]
    windows = np.stack([parse_window(w) for w in df['Window'].values])
    X_ts = extract_ts_features(windows)

    # TF-IDF for text
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    X_txt = tfidf.fit_transform(texts)

    # Stratified 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_splits = skf.get_n_splits()

    # Model definitions
    def get_text_clf():
        return LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)

    def get_ts_mlp():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
        ])

    def get_svd_mlp_fusion():
        return Pipeline([
            ('svd', TruncatedSVD(n_components=128, random_state=42)),
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, random_state=42))
        ])

    def get_rf():
        return RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)

    def get_meta_lr():
        return LogisticRegression(class_weight='balanced', random_state=42)

    # Storage for fold results
    results = {
        'Baseline_Text_Only': [],
        'Baseline_TS_MLP': [],
        'Nonlinear_SVD_MLP_Fusion': [],
        'Entropy_Gated_Fusion': [],
        'Stacking_Meta_Fusion': []
    }

    fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_txt, y)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        X_txt_train, X_txt_test = X_txt[train_idx], X_txt[test_idx]
        X_ts_train, X_ts_test = X_ts[train_idx], X_ts[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_result = {'fold': fold_idx + 1}

        # 1. Baseline_Text_Only
        clf_txt = get_text_clf()
        clf_txt.fit(X_txt_train, y_train)
        pred_txt = clf_txt.predict(X_txt_test)
        f1_txt = f1_score(y_test, pred_txt, average='macro')
        results['Baseline_Text_Only'].append(f1_txt)
        fold_result['Baseline_Text_Only'] = f1_txt
        print(f"  Baseline_Text_Only: {f1_txt:.4f}")

        # 2. Baseline_TS_MLP
        clf_ts = get_ts_mlp()
        clf_ts.fit(X_ts_train, y_train)
        pred_ts = clf_ts.predict(X_ts_test)
        f1_ts = f1_score(y_test, pred_ts, average='macro')
        results['Baseline_TS_MLP'].append(f1_ts)
        fold_result['Baseline_TS_MLP'] = f1_ts
        print(f"  Baseline_TS_MLP: {f1_ts:.4f}")

        # 3. Nonlinear_SVD_MLP_Fusion
        # Fit SVD on train TF-IDF
        svd = TruncatedSVD(n_components=128, random_state=42)
        X_txt_train_svd = svd.fit_transform(X_txt_train)
        X_txt_test_svd = svd.transform(X_txt_test)
        # Concat with TS features
        X_fusion_train = np.hstack([X_txt_train_svd, X_ts_train])
        X_fusion_test = np.hstack([X_txt_test_svd, X_ts_test])
        clf_fusion = get_svd_mlp_fusion()
        clf_fusion.fit(X_fusion_train, y_train)
        pred_fusion = clf_fusion.predict(X_fusion_test)
        f1_fusion = f1_score(y_test, pred_fusion, average='macro')
        results['Nonlinear_SVD_MLP_Fusion'].append(f1_fusion)
        fold_result['Nonlinear_SVD_MLP_Fusion'] = f1_fusion
        print(f"  Nonlinear_SVD_MLP_Fusion: {f1_fusion:.4f}")

        # 4. Entropy_Gated_Fusion
        # Train RF on text (TF-IDF) and TS separately
        rf_txt = get_rf()
        rf_ts = get_rf()
        rf_txt.fit(X_txt_train, y_train)
        rf_ts.fit(X_ts_train, y_train)
        proba_txt = rf_txt.predict_proba(X_txt_test)
        proba_ts = rf_ts.predict_proba(X_ts_test)
        # Entropy weights
        H_txt = compute_entropy(proba_txt)
        H_ts = compute_entropy(proba_ts)
        w_txt = 1.0 / (H_txt + 0.1)
        w_ts = 1.0 / (H_ts + 0.1)
        w_sum = w_txt + w_ts
        w_txt = w_txt / w_sum
        w_ts = w_ts / w_sum
        # Weighted probability fusion
        proba_fused = w_txt[:, None] * proba_txt + w_ts[:, None] * proba_ts
        pred_entropy = np.argmax(proba_fused, axis=1)
        f1_entropy = f1_score(y_test, pred_entropy, average='macro')
        results['Entropy_Gated_Fusion'].append(f1_entropy)
        fold_result['Entropy_Gated_Fusion'] = f1_entropy
        print(f"  Entropy_Gated_Fusion: {f1_entropy:.4f}")

        # 5. Stacking_Meta_Fusion
        # Inner 3-fold CV on training data for out-of-fold probabilities
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # Text classifier (LogisticRegression on TF-IDF)
        clf_txt_inner = get_text_clf()
        # TS classifier (RandomForest on TS features)
        clf_ts_inner = get_rf()
        # Get out-of-fold predictions on train
        oof_txt = cross_val_predict(clf_txt_inner, X_txt_train, y_train, cv=inner_cv, method='predict_proba', n_jobs=-1)
        oof_ts = cross_val_predict(clf_ts_inner, X_ts_train, y_train, cv=inner_cv, method='predict_proba', n_jobs=-1)
        # Interaction features
        oof_interaction = oof_txt * oof_ts
        X_meta_train = np.hstack([oof_txt, oof_ts, oof_interaction])
        # Train meta-learner
        meta_clf = get_meta_lr()
        meta_clf.fit(X_meta_train, y_train)
        # Predict on test: need probabilities from classifiers trained on full train
        clf_txt_full = get_text_clf()
        clf_ts_full = get_rf()
        clf_txt_full.fit(X_txt_train, y_train)
        clf_ts_full.fit(X_ts_train, y_train)
        proba_txt_test = clf_txt_full.predict_proba(X_txt_test)
        proba_ts_test = clf_ts_full.predict_proba(X_ts_test)
        proba_interaction_test = proba_txt_test * proba_ts_test
        X_meta_test = np.hstack([proba_txt_test, proba_ts_test, proba_interaction_test])
        pred_meta = meta_clf.predict(X_meta_test)
        f1_meta = f1_score(y_test, pred_meta, average='macro')
        results['Stacking_Meta_Fusion'].append(f1_meta)
        fold_result['Stacking_Meta_Fusion'] = f1_meta
        print(f"  Stacking_Meta_Fusion: {f1_meta:.4f}")

        fold_details.append(fold_result)

    # Aggregate results
    summary = {}
    for model_name, scores in results.items():
        scores_arr = np.array(scores)
        summary[model_name] = {
            'fold_scores': scores,
            'mean': float(scores_arr.mean()),
            'std': float(scores_arr.std(ddof=1))
        }
        print(f"\n{model_name}: {scores_arr.mean():.4f} +/- {scores_arr.std(ddof=1):.4f}")

    # Paired t-tests vs Baseline_Text_Only
    baseline_scores = np.array(results['Baseline_Text_Only'])
    t_tests = {}
    for model_name in ['Baseline_TS_MLP', 'Nonlinear_SVD_MLP_Fusion', 'Entropy_Gated_Fusion', 'Stacking_Meta_Fusion']:
        model_scores = np.array(results[model_name])
        t_stat, p_val = ttest_rel(model_scores, baseline_scores)
        t_tests[model_name] = {'t_statistic': float(t_stat), 'p_value': float(p_val)}
        print(f"  t-test vs Baseline_Text_Only: t={t_stat:.4f}, p={p_val:.4f}")

    # Save results
    output = {
        'fold_details': fold_details,
        'summary': summary,
        'paired_ttest_vs_baseline_text_only': t_tests
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()