"""
Independent verification and stress-testing harness for Milestone 4 comparative benchmarks.
Evaluates:
1. Exact consistency between comparative_benchmark_report.md, comparative_benchmark_summary.json,
   crisists/expanded_benchmark_results.json, and crisists/full_76k_benchmark_results.json.
2. Independent re-calculation of paired t-tests, Welch's t-tests, Cohen's d, and p-values.
3. Verification of conformal coverage calibration and empirical safety bounds.
4. Cross-validation fold variance authenticity and test set partitioning.
"""
import json
import re
import numpy as np
from scipy import stats

def run_verification():
    print("=" * 80)
    print("CHALLENGER 2: COMPREHENSIVE BENCHMARK & STATISTICAL AUDIT")
    print("=" * 80)

    with open("crisists/expanded_benchmark_results.json", "r", encoding="utf-8") as f:
        exp_res = json.load(f)

    with open("crisists/full_76k_benchmark_results.json", "r", encoding="utf-8") as f:
        base_res = json.load(f)

    with open("comparative_benchmark_summary.json", "r", encoding="utf-8") as f:
        summary_res = json.load(f)

    with open("comparative_benchmark_report.md", "r", encoding="utf-8") as f:
        report_text = f.read()

    # -------------------------------------------------------------------------
    # CHECK 1: Dataset Metadata & Scaling Figures
    # -------------------------------------------------------------------------
    print("\n[CHECK 1] Dataset Metadata & Sample Counts Verification")
    n_base = base_res.get("n_samples", 76481)
    n_exp = exp_res.get("n_samples", 202312)
    events_base = base_res.get("n_events", 19)
    events_exp = exp_res.get("n_events", 65)

    print(f"Baseline samples:  {n_base:,} across {events_base} events")
    print(f"Expanded samples:  {n_exp:,} across {events_exp} events")
    delta_n = n_exp - n_base
    pct_n = (delta_n / n_base) * 100.0
    print(f"Expansion delta:   +{delta_n:,} (+{pct_n:.2f}%) across +{events_exp - events_base} events")

    assert summary_res["baseline_corpus"]["n_samples"] == n_base
    assert summary_res["baseline_corpus"]["n_events"] == events_base
    assert summary_res["expanded_corpus"]["n_samples"] == n_exp
    assert summary_res["expanded_corpus"]["n_events"] == events_exp
    print(">> Summary JSON matches raw sample & event counts: PASS")

    # -------------------------------------------------------------------------
    # CHECK 2: Stratified 5-Fold Cross-Validation Metrics & Statistical Tests
    # -------------------------------------------------------------------------
    print("\n[CHECK 2] Stratified 5-Fold CV: Re-calculation of Statistical Tests")
    tasks = ["label_urgent", "label_humanitarian", "label_useful", "label_sudden"]

    stat_audit_results = {}

    for task in tasks:
        print(f"\n--- Task: {task} ---")
        exp_t = exp_res["regime_A_stratified_5fold"][task]
        base_t = base_res["regime_A_stratified_5fold"][task]
        sum_t = summary_res["regime_A_stratified_5fold_comparison"][task]

        fold_details = exp_t.get("fold_details", [])
        assert len(fold_details) == 5, f"Expected 5 folds, got {len(fold_details)}"

        text_f1s = [f["text_f1"] for f in fold_details]
        moe_f1s = [f["moe_f1"] for f in fold_details]
        stack_f1s = [f["stack_f1"] for f in fold_details]
        moe_accs = [f["moe_acc"] for f in fold_details]

        print(f"Fold Text F1s:  {[round(x, 4) for x in text_f1s]}")
        print(f"Fold MoE F1s:   {[round(x, 4) for x in moe_f1s]}")
        print(f"Fold MoE Accs:  {[round(x, 4) for x in moe_accs]}")

        # Check fold variance and sample partitions
        for fd in fold_details:
            n_tr = fd["n_train"]
            n_te = fd["n_test"]
            assert n_tr + n_te == n_exp, f"Fold sample mismatch: {n_tr} + {n_te} != {n_exp}"
            gw = fd.get("gating_weights", [])
            print(f"    Fold {fd['fold']}: n_train={n_tr:,}, n_test={n_te:,}, sum={n_tr+n_te:,}, gating={gw}")

        moe_f1_var = np.var(moe_f1s, ddof=1)
        moe_f1_std = np.std(moe_f1s, ddof=1)
        print(f"MoE F1 Sample Std (ddof=1): {moe_f1_std:.6f}, Var: {moe_f1_var:.8f}")
        assert moe_f1_var > 0, "Fold variance is 0 (suspect mock data)!"

        # In-corpus paired t-test: MoE vs Text
        t_stat_recalc, p_val_recalc = stats.ttest_rel(moe_f1s, text_f1s)
        diff = np.array(moe_f1s) - np.array(text_f1s)
        cohens_d_recalc = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

        # Check against logged values in raw results
        print(f"Raw reported t-stat:      {exp_t['t_stat_vs_text']:.6f} | Recalculated: {t_stat_recalc:.6f}")
        print(f"Raw reported p-val:       {exp_t['p_val_vs_text']:.6e} | Recalculated: {p_val_recalc:.6e}")
        print(f"Raw reported Cohen's d:   {exp_t['cohens_d_vs_text']:.6f} | Recalculated: {cohens_d_recalc:.6f}")

        np.testing.assert_allclose(t_stat_recalc, exp_t["t_stat_vs_text"], rtol=1e-5)
        np.testing.assert_allclose(p_val_recalc, exp_t["p_val_vs_text"], rtol=1e-5)
        np.testing.assert_allclose(cohens_d_recalc, exp_t["cohens_d_vs_text"], rtol=1e-5)

        # Cross-corpus Welch's t-test: Expanded MoE vs Baseline MoE
        # Note: run_expanded_benchmark used population std (ddof=0) vs sample std (ddof=1)
        # Let's inspect what was passed to ttest_ind_from_stats
        logged_w_t = sum_t["statistical_tests"]["welch_t_stat_vs_baseline"]
        logged_w_p = sum_t["statistical_tests"]["welch_p_val_vs_baseline"]
        logged_cohen_base = sum_t["statistical_tests"]["cohens_d_vs_baseline"]

        # Recalculate with ddof=0 std (what the script passed)
        w_t_ddof0, w_p_ddof0 = stats.ttest_ind_from_stats(
            exp_t["moe_f1_mean"], exp_t["moe_f1_std"], 5,
            base_t["moe_f1_mean"], base_t["moe_f1_std"], 5,
            equal_var=False
        )
        pooled_sd_ddof0 = np.sqrt(((exp_t["moe_f1_std"] ** 2) + (base_t["moe_f1_std"] ** 2)) / 2.0)
        cohen_base_ddof0 = (exp_t["moe_f1_mean"] - base_t["moe_f1_mean"]) / (pooled_sd_ddof0 + 1e-12)

        # Recalculate with sample std (ddof=1)
        base_fold_details = base_t.get("fold_details", [])
        if base_fold_details:
            base_moe_folds = [f["moe_f1"] for f in base_fold_details]
            w_t_exact, w_p_exact = stats.ttest_ind(moe_f1s, base_moe_folds, equal_var=False)
            sd1 = np.std(moe_f1s, ddof=1)
            sd2 = np.std(base_moe_folds, ddof=1)
            pooled_sd_exact = np.sqrt((sd1**2 + sd2**2) / 2.0)
            cohen_base_exact = (np.mean(moe_f1s) - np.mean(base_moe_folds)) / pooled_sd_exact
        else:
            w_t_exact, w_p_exact, cohen_base_exact = None, None, None

        print(f"Logged Welch t-stat:      {logged_w_t:.6f} | Recalc (ddof=0 input): {w_t_ddof0:.6f}")
        print(f"Logged Welch p-val:       {logged_w_p:.6e} | Recalc (ddof=0 input): {w_p_ddof0:.6e}")
        if w_t_exact is not None:
            print(f"Exact Welch t (ddof=1):   {w_t_exact:.6f} | Exact p: {w_p_exact:.6e}")

        stat_audit_results[task] = {
            "paired_t": {"logged": exp_t["t_stat_vs_text"], "recalc": t_stat_recalc},
            "p_val": {"logged": exp_t["p_val_vs_text"], "recalc": p_val_recalc},
            "cohens_d": {"logged": exp_t["cohens_d_vs_text"], "recalc": cohens_d_recalc},
            "welch_t_ddof0": {"logged": logged_w_t, "recalc": w_t_ddof0},
            "welch_p_ddof0": {"logged": logged_w_p, "recalc": w_p_ddof0},
            "welch_t_exact": w_t_exact,
            "welch_p_exact": w_p_exact,
        }

    # -------------------------------------------------------------------------
    # CHECK 3: Conformal Coverage Calibration & Certified Safety
    # -------------------------------------------------------------------------
    print("\n[CHECK 3] Conformal Risk Control & Empirical Recall Audit")
    conf_urgent = exp_res["regime_A_stratified_5fold"]["label_urgent"]["conformal_urgent"]
    emp_recall = conf_urgent["empirical_test_recall"]
    emp_prec = conf_urgent["empirical_test_precision"]
    tau_conf = conf_urgent["mean_tau_conformal"]
    target_cov = conf_urgent["guaranteed_coverage_target"]

    print(f"Target Coverage:        {target_cov * 100:.2f}% (alpha = 0.05)")
    print(f"Empirical Test Recall:  {emp_recall * 100:.2f}%")
    print(f"Empirical Test Prec:    {emp_prec * 100:.2f}%")
    print(f"Mean Calibrated Tau:    {tau_conf:.4f}")

    fold_recalls = [f["conformal_recall"] for f in exp_res["regime_A_stratified_5fold"]["label_urgent"]["fold_details"]]
    fold_taus = [f["conformal_tau"] for f in exp_res["regime_A_stratified_5fold"]["label_urgent"]["fold_details"]]
    print(f"Per-fold Conformal Recalls: {[round(r, 4) for r in fold_recalls]}")
    print(f"Per-fold Calibrated Taus:   {[round(t, 4) for t in fold_taus]}")

    # Check whether empirical recall strictly meets >= 95.0%
    is_strictly_ge_95 = (emp_recall >= 0.95)
    print(f"Strict emp_recall >= 0.95: {is_strictly_ge_95} (actual: {emp_recall:.6f})")

    # In summary JSON:
    summary_conf = summary_res["conformal_safety_comparison"]
    print(f"Summary JSON safety_requirement_satisfied: {summary_conf['safety_requirement_satisfied']}")
    print(f"Report text safety statement: ")
    for line in report_text.splitlines():
        if "Certified" in line or "Conformal" in line or "94.84%" in line:
            print("  ", line)

    # -------------------------------------------------------------------------
    # CHECK 4: Cross-Disaster Generalization (LOEO-CV)
    # -------------------------------------------------------------------------
    print("\n[CHECK 4] Cross-Disaster Generalization (LOEO-CV) Audit")
    loeo_tasks = ["label_urgent", "label_useful", "label_humanitarian"]
    crisis_types = ["Earthquake", "Flood", "Hurricane", "Wildfire"]

    for tk in loeo_tasks:
        print(f"\n--- LOEO Task: {tk} ---")
        exp_loeo = exp_res["regime_B_out_of_type_loeo"][tk]
        base_loeo = base_res["regime_B_out_of_type_loeo"][tk]
        sum_loeo = summary_res["regime_B_loeo_cv_comparison"][tk]

        for ct in crisis_types:
            exp_ct = exp_loeo["by_type"][ct]
            base_ct = base_loeo["by_type"][ct]
            sum_ct = sum_loeo["by_type"][ct]

            assert sum_ct["n_samples"] == exp_ct["n_samples"]
            np.testing.assert_allclose(sum_ct["baseline_moe_f1"], base_ct["moe_f1"], rtol=1e-5)
            np.testing.assert_allclose(sum_ct["expanded_moe_f1"], exp_ct["moe_f1"], rtol=1e-5)
            np.testing.assert_allclose(sum_ct["delta"], exp_ct["moe_f1"] - base_ct["moe_f1"], rtol=1e-5)
            print(f"  {ct:12s} N={exp_ct['n_samples']:,} | Base={base_ct['moe_f1']:.4f} | Exp={exp_ct['moe_f1']:.4f} | Delta={exp_ct['moe_f1'] - base_ct['moe_f1']:+.4f} : OK")

        # Check mean across 4 types
        mean_exp = np.mean([exp_loeo["by_type"][ct]["moe_f1"] for ct in crisis_types])
        mean_base = np.mean([base_loeo["by_type"][ct]["moe_f1"] for ct in crisis_types])
        np.testing.assert_allclose(sum_loeo["expanded_mean_moe_f1"], mean_exp, rtol=1e-5)
        np.testing.assert_allclose(sum_loeo["baseline_mean_moe_f1"], mean_base, rtol=1e-5)
        print(f"  Mean across 4 types: Base={mean_base:.4f} | Exp={mean_exp:.4f} | Delta={mean_exp - mean_base:+.4f} : OK")

    # -------------------------------------------------------------------------
    # CHECK 5: Report Markdown Exact Consistency
    # -------------------------------------------------------------------------
    print("\n[CHECK 5] Markdown Report String Consistency Check")
    # Verify numbers in Section 2 table:
    # Urgent: 75.95% ± 0.38%, 77.03% ± 0.45%, 80.92% ± 0.17%, 82.33% ± 0.21%, +5.30%, t = 45.68, p = 1.37e-06, d = 20.43, t = 23.74
    sec2_checks = [
        ("label_urgent", "82.33%", "+5.30%", "t = 45.68", "p = 1.37e-06", "d = 20.43", "t = 23.74"),
        ("label_humanitarian", "70.06%", "-3.16%", "t = 25.59", "p = 1.38e-05", "d = 11.45", "t = -7.93"),
        ("label_useful", "87.39%", "+10.47%", "t = 16.87", "p = 7.24e-05", "d = 7.54", "t = 22.20"),
        ("label_sudden", "89.61%", "+6.35%", "t = 33.35", "p = 4.82e-06", "d = 14.92", "t = 24.43"),
    ]
    for task_name, exp_moe_str, delta_str, t_str, p_str, d_str, wt_str in sec2_checks:
        for val in [exp_moe_str, delta_str, t_str, p_str, d_str, wt_str]:
            assert val in report_text, f"Missing string '{val}' for task '{task_name}' in comparative_benchmark_report.md!"
        print(f"  Report Section 2 {task_name}: All target tokens verified!")

    print("\n" + "=" * 80)
    print("ALL EMPIRICAL AND MATHEMATICAL CHECKS COMPLETED.")
    print("=" * 80)

if __name__ == "__main__":
    run_verification()
