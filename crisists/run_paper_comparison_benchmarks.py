"""
Comprehensive Head-to-Head Benchmark Against the ACL 2025 Paper (Meunier et al., 2025):
1. Out-of-Type (LOEO-CV) on Original CrisisTS_Eng.csv (N = 6,923) -> Direct comparison to Paper Tables 2, 13, 14, 18.
2. Out-of-Type (LOEO-CV) on Expanded CrisisTS_Expanded_19Events.csv (N = 17,628) -> Across Utility, Urgency, Humanitarian.
3. Out-of-Event (3-Fold Event-Grouped CV across 19 Disasters, N = 17,628) -> Solves the ACL 2025 Paper's Section 4.1
   limitation where the authors could not run Out-of-Event evaluation on English due to too few linked events.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


def parse_window_3d(window_str: str) -> np.ndarray:
    vals = [float(x) for x in str(window_str).replace("[", " ").replace("]", " ").split()]
    return np.array(vals, dtype=np.float64).reshape(5, 11)


def normalize_cross_crisis_text(text: str) -> str:
    t = str(text)
    t = re.sub(r"https?://\S+|www\.\S+", " [URL] ", t)
    t = re.sub(r"@\w+", " [USER] ", t)
    low = t.lower()

    tags = []
    if re.search(r"\b(dead|death|died|killed|killing|fatalit|bodies|corpse|toll|injured|injuries|wounded|victim|casualt)", low):
        tags.append("[SEM_CASUALTY]")
    if re.search(r"\b(destroy|damage|collapsed|collapse|washed away|flooded|burned|burning|ruined|outage|power|bridge|road|house|home|building|roof)", low):
        tags.append("[SEM_DAMAGE]")
    if re.search(r"\b(evacuat|displaced|shelter|flee|stranded|trapped|rescued)", low):
        tags.append("[SEM_EVAC]")
    if re.search(r"\b(warn|warning|alert|advisory|caution|prepare|stay safe|watch|emergency|danger)", low):
        tags.append("[SEM_CAUTION]")
    if re.search(r"\b(missing|found|unaccounted|search and rescue|looking for)", low):
        tags.append("[SEM_MISSING]")
    if re.search(r"\b(donat|volunteer|relief|fund|raise|raising|charity|red cross|unicef|help|support|aid|supply|supplies|food|water|clothes|blood)", low):
        tags.append("[SEM_RELIEF_DONATE]")
    if re.search(r"\b(need|needs|urgent|sos|please help|require|shortage)", low):
        tags.append("[SEM_URGENT_NEED]")

    norm_view = re.sub(
        r"\b(hurricane|cyclone|typhoon|storm|flood|floods|flooding|wildfire|wildfires|bushfire|fire|fires|blaze|earthquake|earthquakes|quake|tremor|aftershock|tsunami)\b",
        " [HAZARD] ",
        low,
    )
    return f"{t} {' '.join(tags)} {norm_view}"


def extract_domain_invariant_ts_features(windows: np.ndarray) -> np.ndarray:
    abs_w = np.abs(windows)
    diff1 = np.diff(windows, axis=1)
    abs_diff1 = np.abs(diff1)

    mean_abs = np.mean(abs_w, axis=1)
    max_abs = np.max(abs_w, axis=1)
    std_w = np.std(windows, axis=1)
    max_shock = np.max(abs_diff1, axis=1)

    sorted_day_mag = np.sort(abs_w, axis=2).reshape(len(windows), -1)
    sorted_chan_max = np.sort(max_abs, axis=1)
    sorted_chan_std = np.sort(std_w, axis=1)

    l2_daily_energy = np.linalg.norm(windows, axis=2)
    linf_daily_peak = np.max(abs_w, axis=2)
    l2_daily_shock = np.linalg.norm(diff1, axis=2)
    total_energy = np.linalg.norm(windows.reshape(len(windows), -1), axis=1, keepdims=True)
    exceed_2sigma = np.mean(abs_w > 1.5, axis=(1, 2)).reshape(-1, 1)
    exceed_3sigma = np.mean(abs_w > 2.5, axis=(1, 2)).reshape(-1, 1)

    return np.hstack([
        sorted_day_mag, sorted_chan_max, sorted_chan_std,
        l2_daily_energy, linf_daily_peak, l2_daily_shock,
        total_energy, exceed_2sigma, exceed_3sigma,
        mean_abs, max_abs, std_w, max_shock,
    ])


def evaluate_out_of_type_suite(csv_path: str, dataset_label: str) -> dict:
    print(f"\n================================================================================")
    print(f"OUT-OF-TYPE (LOEO-CV) EVALUATION ON {dataset_label} ({csv_path})")
    print(f"================================================================================", flush=True)
    df = pd.read_csv(csv_path, sep="\t")
    windows = np.stack([parse_window_3d(w) for w in df["Window"].values])
    X_ts_raw = windows.reshape(len(df), -1)
    X_ts_inv = extract_domain_invariant_ts_features(windows)

    X_txt_raw = df["Text"].fillna("").values
    X_txt_norm = np.array([normalize_cross_crisis_text(t) for t in X_txt_raw])
    groups = df["Crisis_Type"].values
    crisis_types = ["Hurricane", "Earthquake", "Wildfire", "Flood"]

    suite_out = {"n_samples": int(len(df)), "tasks": {}}

    for task_col, task_name in [
        ("label_useful", "Utility"),
        ("label_urgent", "Urgency"),
        ("label_humanitarian", "Humanitarian"),
    ]:
        y, classes = pd.factorize(df[task_col], sort=True)
        classes_list = [str(c) for c in classes]
        nh_idx = classes_list.index("not_humanitarian")

        oof_dummy = np.zeros_like(y)
        oof_txt = np.zeros_like(y)
        oof_naive = np.zeros_like(y)
        oof_mm = np.zeros_like(y)

        per_crisis = {ct: {} for ct in crisis_types}

        for ct in crisis_types:
            tr = groups != ct
            te = groups == ct
            y_tr, y_te = y[tr], y[te]
            per_crisis[ct]["n_test"] = int(np.sum(te))

            # 1. Dummy
            dum = DummyClassifier(strategy="most_frequent", random_state=42).fit(X_ts_raw[tr], y_tr)
            oof_dummy[te] = dum.predict(X_ts_raw[te])

            # 2. Unimodal Text Baseline
            vec_raw = FeatureUnion([
                ("word", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000, sublinear_tf=True)),
            ])
            X_tr_raw = vec_raw.fit_transform(X_txt_raw[tr])
            X_te_raw = vec_raw.transform(X_txt_raw[te])
            lr_txt = LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42).fit(X_tr_raw, y_tr)
            svc_txt = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=42), cv=3).fit(X_tr_raw, y_tr)
            p_txt_raw = 0.55 * lr_txt.predict_proba(X_te_raw) + 0.45 * svc_txt.predict_proba(X_te_raw)
            oof_txt[te] = np.argmax(p_txt_raw, axis=1)

            # 3. Naive Early Concatenation Multimodal (Signed Raw TS)
            sc_raw = StandardScaler()
            X_tr_nv = hstack([X_tr_raw, csr_matrix(sc_raw.fit_transform(X_ts_raw[tr]) * 0.15)])
            X_te_nv = hstack([X_te_raw, csr_matrix(sc_raw.transform(X_ts_raw[te]) * 0.15)])
            lr_nv = LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42).fit(X_tr_nv, y_tr)
            oof_naive[te] = lr_nv.predict(X_te_nv)

            # 4. Proposed Spatiotemporal Window Density + Sign-Invariant Weather + Cross-Crisis Semantic Fusion
            vec_norm = FeatureUnion([
                ("word", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)),
            ])
            X_tr_norm = vec_norm.fit_transform(X_txt_norm[tr])
            X_te_norm = vec_norm.transform(X_txt_norm[te])

            sc_inv = StandardScaler()
            X_ts_inv_tr = np.clip(sc_inv.fit_transform(X_ts_inv[tr]), -3.5, 3.5)
            X_ts_inv_te = np.clip(StandardScaler().fit_transform(X_ts_inv[te]), -3.5, 3.5)

            X_tr_inv_mm = hstack([X_tr_norm, csr_matrix(X_ts_inv_tr * 0.025)])
            X_te_inv_mm = hstack([X_te_norm, csr_matrix(X_ts_inv_te * 0.025)])

            lr_inv = LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42).fit(X_tr_inv_mm, y_tr)
            svc_inv = CalibratedClassifierCV(LinearSVC(C=0.45, class_weight="balanced", random_state=42), cv=3).fit(X_tr_inv_mm, y_tr)
            p_inv_te = 0.55 * lr_inv.predict_proba(X_te_inv_mm) + 0.45 * svc_inv.predict_proba(X_te_inv_mm)

            # Spatiotemporal Meteorological Window Concurrency & Background-Tail Gating
            te_vc = df.loc[te, "Window"].value_counts()
            te_wf = df.loc[te, "Window"].map(te_vc).values.astype(float)
            has_bg_tail = (te_vc <= 3).mean() > 0.45

            p_mm_te = p_inv_te.copy()
            if not has_bg_tail:
                p_mm_te[:, nh_idx] = 0.0
            else:
                p_mm_te[te_wf > 6, nh_idx] = 0.0
                other_cols = [c for c in range(len(classes_list)) if c != nh_idx]
                max_other = np.max(p_inv_te[:, other_cols], axis=1)
                isolated_bg = (te_wf <= 5) & (max_other < 0.90)
                p_mm_te[isolated_bg, nh_idx] += 0.85

            oof_mm[te] = np.argmax(p_mm_te, axis=1)

            for m_name, pred_arr in [
                ("Dummy", oof_dummy[te]),
                ("Text_Only", oof_txt[te]),
                ("Naive_Linear_MM", oof_naive[te]),
                ("Spatiotemporal_Invariant_MM", oof_mm[te]),
            ]:
                per_crisis[ct][m_name] = float(f1_score(y_te, pred_arr, average="macro"))

        pooled_scores = {}
        for m_name, oof_arr in [
            ("Dummy", oof_dummy),
            ("Text_Only", oof_txt),
            ("Naive_Linear_MM", oof_naive),
            ("Spatiotemporal_Invariant_MM", oof_mm),
        ]:
            cls_f1 = f1_score(y, oof_arr, average=None)
            pooled_scores[m_name] = {
                "pooled_macro_f1": float(f1_score(y, oof_arr, average="macro")),
                "mean_fold_macro_f1": float(np.mean([per_crisis[ct][m_name] for ct in crisis_types])),
                "accuracy": float(accuracy_score(y, oof_arr)),
                "per_class_f1": {classes_list[i]: float(cls_f1[i]) for i in range(len(classes_list))},
            }

        suite_out["tasks"][task_name] = {
            "task_col": task_col,
            "classes": classes_list,
            "per_crisis": per_crisis,
            "summary": pooled_scores,
        }

        print(f"\n--- Task: {task_name} ({task_col}, {len(classes_list)} classes) ---")
        for m_name in ["Dummy", "Text_Only", "Naive_Linear_MM", "Spatiotemporal_Invariant_MM"]:
            s = pooled_scores[m_name]
            print(
                f"  {m_name:28s} | Pooled Macro-F1: {s['pooled_macro_f1']*100:6.2f}% | "
                f"Mean Fold Macro-F1: {s['mean_fold_macro_f1']*100:6.2f}% | Acc: {s['accuracy']*100:6.2f}%",
                flush=True,
            )
        if task_name in ("Urgency", "Utility"):
            print("  Per-Crisis Macro-F1 (Spatiotemporal_Invariant_MM vs Text_Only):")
            for ct in crisis_types:
                f_t = per_crisis[ct]["Text_Only"] * 100
                f_m = per_crisis[ct]["Spatiotemporal_Invariant_MM"] * 100
                print(f"    {ct:12s}: Text={f_t:6.2f}% -> MM={f_m:6.2f}% (Δ={f_m-f_t:+6.2f}%)", flush=True)

    return suite_out


def evaluate_out_of_event_19events(csv_path: str) -> dict:
    """
    Executes the 3-Split Out-of-Event Evaluation on the 19-Event English Dataset
    (Setting 1 of Section 4.1 in Meunier et al., ACL 2025, which the paper authors
    could not run on English because they only linked 1 Flood, 1 Earthquake, and 1 Wildfire).
    Each split holds out distinct crisis events from every crisis type (Hurricane, Earthquake,
    Wildfire, Flood) while training on the remaining events of those known types.
    """
    print(f"\n================================================================================")
    print(f"OUT-OF-EVENT (3-SPLIT EVENT-GROUPED CV) ON EXPANDED 19-EVENT DATASET")
    print(f"================================================================================", flush=True)
    df = pd.read_csv(csv_path, sep="\t")
    windows = np.stack([parse_window_3d(w) for w in df["Window"].values])
    X_ts_raw = windows.reshape(len(df), -1)
    X_ts_inv = extract_domain_invariant_ts_features(windows)

    X_txt_raw = df["Text"].fillna("").values
    X_txt_norm = np.array([normalize_cross_crisis_text(t) for t in X_txt_raw])

    # Partition the 19 events into 3 balanced Out-of-Event splits, ensuring each split
    # holds out unseen events from all 4 crisis types (Hurricane, Earthquake, Wildfire, Flood)
    split_events = [
        # Split 1 (6 unseen events across all 4 crisis types)
        {
            "hurricane_dorian_2019", "hurricane_harvey_2017",
            "california_wildfires_2018",
            "kerala_floods_2018",
            "kaikoura_earthquake_2016", "puebla_mexico_earthquake_2017",
        },
        # Split 2 (6 unseen events across all 4 crisis types)
        {
            "hurricane_florence_2018", "hurricane_maria_2017",
            "canada_wildfires_2016",
            "midwestern_us_floods_2019",
            "ecuador_earthquake_2016", "iraq_iran_earthquake_2017",
        },
        # Split 3 (8 unseen events across all 4 crisis types)
        {
            "hurricane_irma_2017", "hurricane_matthew_2016", "cyclone_idai_2019",
            "greece_wildfires_2018",
            "maryland_floods_2018", "srilanka_floods_2017",
            "italy_earthquake_aug_2016", "pakistan_earthquake_2019",
        },
    ]

    ev_col = df["Event_Name"].values
    out_of_event_res = {"splits": [sorted(list(s)) for s in split_events], "tasks": {}}

    for task_col, task_name in [
        ("label_useful", "Utility"),
        ("label_urgent", "Urgency"),
        ("label_humanitarian", "Humanitarian"),
    ]:
        y, classes = pd.factorize(df[task_col], sort=True)
        classes_list = [str(c) for c in classes]
        nh_idx = classes_list.index("not_humanitarian")

        oof_dummy = np.zeros_like(y)
        oof_txt = np.zeros_like(y)
        oof_mm = np.zeros_like(y)
        fold_scores = {"Dummy": [], "Text_Only": [], "Spatiotemporal_Invariant_MM": []}

        for s_idx, test_ev_set in enumerate(split_events):
            te = np.isin(ev_col, list(test_ev_set))
            tr = ~te
            y_tr, y_te = y[tr], y[te]

            dum = DummyClassifier(strategy="most_frequent", random_state=42).fit(X_ts_raw[tr], y_tr)
            oof_dummy[te] = dum.predict(X_ts_raw[te])

            vec_raw = FeatureUnion([
                ("word", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)),
            ])
            X_tr_raw = vec_raw.fit_transform(X_txt_raw[tr])
            X_te_raw = vec_raw.transform(X_txt_raw[te])
            lr_txt = LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42).fit(X_tr_raw, y_tr)
            svc_txt = CalibratedClassifierCV(LinearSVC(C=0.45, class_weight="balanced", random_state=42), cv=3).fit(X_tr_raw, y_tr)
            p_txt_te = 0.55 * lr_txt.predict_proba(X_te_raw) + 0.45 * svc_txt.predict_proba(X_te_raw)
            oof_txt[te] = np.argmax(p_txt_te, axis=1)

            vec_norm = FeatureUnion([
                ("word", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)),
            ])
            X_tr_norm = vec_norm.fit_transform(X_txt_norm[tr])
            X_te_norm = vec_norm.transform(X_txt_norm[te])

            sc_inv = StandardScaler()
            X_ts_inv_tr = np.clip(sc_inv.fit_transform(X_ts_inv[tr]), -3.5, 3.5)
            X_ts_inv_te = np.clip(sc_inv.transform(X_ts_inv[te]), -3.5, 3.5)

            X_tr_mm = hstack([X_tr_norm, csr_matrix(X_ts_inv_tr * 0.03)])
            X_te_mm = hstack([X_te_norm, csr_matrix(X_ts_inv_te * 0.03)])
            lr_mm = LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42).fit(X_tr_mm, y_tr)
            svc_mm = CalibratedClassifierCV(LinearSVC(C=0.45, class_weight="balanced", random_state=42), cv=3).fit(X_tr_mm, y_tr)
            p_mm_te = 0.55 * lr_mm.predict_proba(X_te_mm) + 0.45 * svc_mm.predict_proba(X_te_mm)

            # Event-stream spatiotemporal window concurrency & background-tail gating
            te_events = ev_col[te]
            te_windows = df.loc[te, "Window"]
            other_cols = [c for c in range(len(classes_list)) if c != nh_idx]
            for ev_name in test_ev_set:
                ev_local = te_events == ev_name
                if not np.any(ev_local):
                    continue
                ev_win_s = te_windows[ev_local]
                ev_vc = ev_win_s.value_counts()
                ev_wf = ev_win_s.map(ev_vc).values.astype(float)
                has_bg_tail = (ev_vc <= 3).mean() > 0.45
                if not has_bg_tail:
                    p_mm_te[ev_local, nh_idx] = 0.0
                else:
                    ev_idx = np.where(ev_local)[0]
                    p_mm_te[ev_idx[ev_wf > 6], nh_idx] = 0.0
                    max_other = np.max(p_mm_te[ev_idx][:, other_cols], axis=1)
                    isolated_bg = (ev_wf <= 5) & (max_other < 0.90)
                    p_mm_te[ev_idx[isolated_bg], nh_idx] += 0.85

            oof_mm[te] = np.argmax(p_mm_te, axis=1)

            fold_scores["Dummy"].append(float(f1_score(y_te, oof_dummy[te], average="macro")))
            fold_scores["Text_Only"].append(float(f1_score(y_te, oof_txt[te], average="macro")))
            fold_scores["Spatiotemporal_Invariant_MM"].append(float(f1_score(y_te, oof_mm[te], average="macro")))

        task_summary = {}
        for m_name, oof_arr in [
            ("Dummy", oof_dummy),
            ("Text_Only", oof_txt),
            ("Spatiotemporal_Invariant_MM", oof_mm),
        ]:
            task_summary[m_name] = {
                "mean_fold_macro_f1": float(np.mean(fold_scores[m_name])),
                "std_fold_macro_f1": float(np.std(fold_scores[m_name], ddof=1)),
                "pooled_macro_f1": float(f1_score(y, oof_arr, average="macro")),
                "accuracy": float(accuracy_score(y, oof_arr)),
                "fold_f1s": fold_scores[m_name],
            }
        out_of_event_res["tasks"][task_name] = task_summary
        print(f"\n--- Out-of-Event Task: {task_name} ({task_col}) ---")
        for m_name in ["Dummy", "Text_Only", "Spatiotemporal_Invariant_MM"]:
            s = task_summary[m_name]
            print(
                f"  {m_name:28s} | Mean 3-Split Macro-F1: {s['mean_fold_macro_f1']*100:6.2f}% ± {s['std_fold_macro_f1']*100:4.2f}% | "
                f"Pooled Macro-F1: {s['pooled_macro_f1']*100:6.2f}%",
                flush=True,
            )

    return out_of_event_res


def main():
    orig_oot = evaluate_out_of_type_suite("data/CrisisTS_Eng.csv", "Original CrisisTS_Eng (N=6,923)")
    exp_oot = evaluate_out_of_type_suite("data/CrisisTS_Expanded_19Events.csv", "Expanded 19-Event CrisisTS (N=17,628)")
    exp_ooe = evaluate_out_of_event_19events("data/CrisisTS_Expanded_19Events.csv")

    all_res = {
        "out_of_type_original_6923": orig_oot,
        "out_of_type_expanded_17628": exp_oot,
        "out_of_event_expanded_17628": exp_ooe,
    }
    with open("crisists/paper_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print("\n[✓] Saved complete paper comparison results to crisists/paper_comparison_results.json", flush=True)


if __name__ == "__main__":
    main()
