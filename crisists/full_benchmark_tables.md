# Full CrisisTS Benchmark Results (Upgraded Word+Char TF-IDF + 165 TS Features + OOF Stacking)

## label_urgent

| Model | Macro-F1 (Mean ± Std) | Weighted-F1 (Mean ± Std) | Accuracy (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| **B0_Dummy_Majority** | 0.2697 ± 0.0000 | 0.5496 ± 0.0002 | 0.6793 ± 0.0001 |
| **B1_Dummy_Stratified** | 0.3455 ± 0.0057 | 0.5396 ± 0.0078 | 0.5394 ± 0.0078 |
| **B2_TS_HistGB** | 0.7257 ± 0.0079 | 0.7135 ± 0.0069 | 0.7446 ± 0.0052 |
| **B3_TS_RandomForest** | 0.7573 ± 0.0107 | 0.7042 ± 0.0149 | 0.6906 ± 0.0164 |
| **B4_Text_LinearSVC** | 0.7230 ± 0.0326 | 0.8964 ± 0.0110 | 0.9097 ± 0.0094 |
| **B5_Text_LogReg** | 0.8357 ± 0.0212 | 0.9199 ± 0.0115 | 0.9195 ± 0.0116 |
| **M1_Early_Feature_Fusion** | 0.9388 ± 0.0075 | 0.9506 ± 0.0085 | 0.9509 ± 0.0084 |
| **M2_Stacking_Meta_Fusion** | 0.9118 ± 0.0040 | 0.9270 ± 0.0051 | 0.9259 ± 0.0053 |

### Statistical Tests vs. `B5_Text_LogReg`

| Comparison | Δ Macro-F1 | Paired t-stat | t-test p-value | Wilcoxon p-value | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: |
| M1_Early_Feature_Fusion vs B5 | +10.32% | 14.0805 | 0.000148 | 0.0625 | 6.2970 |
| M2_Stacking_Meta_Fusion vs B5 | +7.61% | 8.0406 | 0.001299 | 0.0625 | 3.5959 |


## label_humanitarian

| Model | Macro-F1 (Mean ± Std) | Weighted-F1 (Mean ± Std) | Accuracy (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| **B0_Dummy_Majority** | 0.0523 ± 0.0001 | 0.1107 ± 0.0003 | 0.2646 ± 0.0004 |
| **B1_Dummy_Stratified** | 0.1311 ± 0.0075 | 0.1754 ± 0.0069 | 0.1762 ± 0.0070 |
| **B2_TS_HistGB** | 0.4248 ± 0.0114 | 0.4356 ± 0.0094 | 0.4739 ± 0.0100 |
| **B3_TS_RandomForest** | 0.4367 ± 0.0089 | 0.4095 ± 0.0081 | 0.4101 ± 0.0068 |
| **B4_Text_LinearSVC** | 0.8424 ± 0.0099 | 0.8800 ± 0.0045 | 0.8846 ± 0.0041 |
| **B5_Text_LogReg** | 0.8552 ± 0.0095 | 0.8844 ± 0.0057 | 0.8837 ± 0.0064 |
| **M1_Early_Feature_Fusion** | 0.9012 ± 0.0086 | 0.9127 ± 0.0048 | 0.9126 ± 0.0049 |
| **M2_Stacking_Meta_Fusion** | 0.8490 ± 0.0159 | 0.8751 ± 0.0116 | 0.8706 ± 0.0139 |

### Statistical Tests vs. `B5_Text_LogReg`

| Comparison | Δ Macro-F1 | Paired t-stat | t-test p-value | Wilcoxon p-value | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: |
| M1_Early_Feature_Fusion vs B5 | +4.60% | 27.7847 | 0.000010 | 0.0625 | 12.4257 |
| M2_Stacking_Meta_Fusion vs B5 | -0.62% | -0.6206 | 0.568503 | 1.0000 | -0.2775 |


## label_useful

| Model | Macro-F1 (Mean ± Std) | Weighted-F1 (Mean ± Std) | Accuracy (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| **B0_Dummy_Majority** | 0.4856 ± 0.0001 | 0.9167 ± 0.0005 | 0.9440 ± 0.0004 |
| **B1_Dummy_Stratified** | 0.4987 ± 0.0037 | 0.8925 ± 0.0010 | 0.8911 ± 0.0009 |
| **B2_TS_HistGB** | 0.9944 ± 0.0064 | 0.9988 ± 0.0013 | 0.9988 ± 0.0013 |
| **B3_TS_RandomForest** | 0.9965 ± 0.0061 | 0.9993 ± 0.0013 | 0.9993 ± 0.0013 |
| **B4_Text_LinearSVC** | 0.6543 ± 0.0340 | 0.9396 ± 0.0053 | 0.9529 ± 0.0037 |
| **B5_Text_LogReg** | 0.7879 ± 0.0234 | 0.9534 ± 0.0047 | 0.9516 ± 0.0050 |
| **M1_Early_Feature_Fusion** | 0.9643 ± 0.0119 | 0.9927 ± 0.0023 | 0.9929 ± 0.0022 |
| **M2_Stacking_Meta_Fusion** | 0.9387 ± 0.0114 | 0.9866 ± 0.0027 | 0.9861 ± 0.0029 |

### Statistical Tests vs. `B5_Text_LogReg`

| Comparison | Δ Macro-F1 | Paired t-stat | t-test p-value | Wilcoxon p-value | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: |
| M1_Early_Feature_Fusion vs B5 | +17.64% | 12.7648 | 0.000217 | 0.0625 | 5.7086 |
| M2_Stacking_Meta_Fusion vs B5 | +15.08% | 20.4204 | 0.000034 | 0.0625 | 9.1323 |


## label_sudden

| Model | Macro-F1 (Mean ± Std) | Weighted-F1 (Mean ± Std) | Accuracy (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| **B0_Dummy_Majority** | 0.2993 ± 0.0001 | 0.7313 ± 0.0004 | 0.8145 ± 0.0003 |
| **B1_Dummy_Stratified** | 0.3386 ± 0.0067 | 0.6870 ± 0.0022 | 0.6876 ± 0.0024 |
| **B2_TS_HistGB** | 0.9972 ± 0.0030 | 0.9991 ± 0.0009 | 0.9991 ± 0.0009 |
| **B3_TS_RandomForest** | 0.9977 ± 0.0033 | 0.9993 ± 0.0010 | 0.9993 ± 0.0010 |
| **B4_Text_LinearSVC** | 0.7509 ± 0.0346 | 0.9359 ± 0.0074 | 0.9477 ± 0.0043 |
| **B5_Text_LogReg** | 0.8355 ± 0.0167 | 0.9542 ± 0.0046 | 0.9565 ± 0.0043 |
| **M1_Early_Feature_Fusion** | 0.9852 ± 0.0050 | 0.9956 ± 0.0014 | 0.9957 ± 0.0014 |
| **M2_Stacking_Meta_Fusion** | 0.9896 ± 0.0055 | 0.9968 ± 0.0017 | 0.9968 ± 0.0017 |

### Statistical Tests vs. `B5_Text_LogReg`

| Comparison | Δ Macro-F1 | Paired t-stat | t-test p-value | Wilcoxon p-value | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: |
| M1_Early_Feature_Fusion vs B5 | +14.97% | 24.0696 | 0.000018 | 0.0625 | 10.7643 |
| M2_Stacking_Meta_Fusion vs B5 | +15.41% | 24.5450 | 0.000016 | 0.0625 | 10.9768 |

