"""
Real Stratified 5-Fold Cross-Validation Verification on CrisisTS_Eng.csv
Executes genuine unimodal vs. multimodal baselines (Feature Concatenation & Late Probability Fusion)
and statistical hypothesis tests (Paired t-test, Wilcoxon, Shapiro-Wilk, Cohen's d) for SC203.
"""
import ast
import json
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")


def parse_window(win_str: str) -> np.ndarray:
    cleaned = re.sub(r"\s+", ",", win_str.strip())
    cleaned = re.sub(r"\[\s*,", "[", cleaned)
    cleaned = re.sub(r",\s*\]", "]", cleaned)
    return np.array(ast.literal_eval(cleaned), dtype=np.float64)


def extract_ts_features(windows: np.ndarray) -> np.ndarray:
    N = windows.shape[0]
    flat = windows.reshape(N, -1)
    mean_c = np.mean(windows, axis=1)
    std_c = np.std(windows, axis=1)
    max_c = np.max(windows, axis=1)
    min_c = np.min(windows, axis=1)
    delta_c = windows[:, -1, :] - windows[:, 0, :]
    return np.hstack([flat, mean_c, std_c, max_c, min_c, delta_c])


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
    text = re.sub(r"@\w+", " [USER] ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv("data/CrisisTS_Eng.csv", sep="\t")
    print(f"Loaded CrisisTS_Eng.csv: {len(df)} rows")

    windows = np.stack([parse_window(w) for w in df["Window"]])
    X_ts_raw = extract_ts_features(windows)
    X_text_raw = df["Text"].apply(clean_text).values

    summary_out = {}

    for target_col in ["label_urgent", "label_humanitarian"]:
        print(f"\n==========================================")
        print(f"TARGET: {target_col} ({df[target_col].nunique()} classes)")
        print(f"==========================================")
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].values)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        results = {
            "Baseline_0_Dummy_Majority": [],
            "Baseline_1_TS_Only": [],
            "Baseline_2_Text_Only": [],
            "Proposed_Early_Feature_Fusion": [],
            "Proposed_Late_Prob_Fusion": [],
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_text_raw, y), 1):
            y_train, y_test = y[train_idx], y[test_idx]

            # 0. Dummy Majority
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_ts_raw[train_idx], y_train)
            pred_dummy = dummy.predict(X_ts_raw[test_idx])
            results["Baseline_0_Dummy_Majority"].append(
                f1_score(y_test, pred_dummy, average="macro", zero_division=0)
            )

            # 1. Time Series Only
            scaler = StandardScaler()
            X_ts_tr = scaler.fit_transform(X_ts_raw[train_idx])
            X_ts_te = scaler.transform(X_ts_raw[test_idx])

            clf_ts = RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
            )
            clf_ts.fit(X_ts_tr, y_train)
            pred_ts = clf_ts.predict(X_ts_te)
            proba_ts_te = clf_ts.predict_proba(X_ts_te)
            results["Baseline_1_TS_Only"].append(
                f1_score(y_test, pred_ts, average="macro", zero_division=0)
            )

            # 2. Text Only (TF-IDF + LogisticRegression)
            vec = TfidfVectorizer(
                max_features=15000, ngram_range=(1, 2), sublinear_tf=True
            )
            X_txt_tr = vec.fit_transform(X_text_raw[train_idx])
            X_txt_te = vec.transform(X_text_raw[test_idx])

            clf_txt = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=42
            )
            clf_txt.fit(X_txt_tr, y_train)
            pred_txt = clf_txt.predict(X_txt_te)
            proba_txt_te = clf_txt.predict_proba(X_txt_te)
            results["Baseline_2_Text_Only"].append(
                f1_score(y_test, pred_txt, average="macro", zero_division=0)
            )

            # 3. Early Feature Concatenation Fusion (strictly features, no in-fold prob leakage)
            X_early_tr = hstack([X_txt_tr, csr_matrix(X_ts_tr * 0.1)])
            X_early_te = hstack([X_txt_te, csr_matrix(X_ts_te * 0.1)])
            clf_early = LogisticRegression(
                C=1.5, class_weight="balanced", max_iter=1000, random_state=42
            )
            clf_early.fit(X_early_tr, y_train)
            pred_early = clf_early.predict(X_early_te)
            results["Proposed_Early_Feature_Fusion"].append(
                f1_score(y_test, pred_early, average="macro", zero_division=0)
            )

            # 4. Late Probability Fusion (0.65 * P_text + 0.35 * P_ts on blind test set)
            proba_late_te = 0.65 * proba_txt_te + 0.35 * proba_ts_te
            pred_late = np.argmax(proba_late_te, axis=1)
            results["Proposed_Late_Prob_Fusion"].append(
                f1_score(y_test, pred_late, average="macro", zero_division=0)
            )

        for name, scores in results.items():
            arr = np.array(scores)
            print(
                f"{name:32s} | Folds: {[round(s, 4) for s in scores]} | Mean: {arr.mean():.4f} +/- {arr.std(ddof=1):.4f}"
            )

        for mm_key in ["Proposed_Early_Feature_Fusion", "Proposed_Late_Prob_Fusion"]:
            diff = np.array(results[mm_key]) - np.array(results["Baseline_2_Text_Only"])
            shapiro_stat, shapiro_p = stats.shapiro(diff)
            t_stat, t_p = stats.ttest_rel(results[mm_key], results["Baseline_2_Text_Only"])
            w_stat, w_p = stats.wilcoxon(results[mm_key], results["Baseline_2_Text_Only"])
            cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-12)
            print(
                f"[{mm_key}] Delta: {diff.mean()*100:+.2f}% | Shapiro p={shapiro_p:.4f} | t={t_stat:.4f} (p={t_p:.6f}) | Wilcoxon p={w_p:.4f} | Cohen's d={cohens_d:.4f}"
            )

        summary_out[target_col] = {
            k: [round(float(x), 4) for x in v] for k, v in results.items()
        }

    with open("crisists/verified_5fold_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)


if __name__ == "__main__":
    main()
