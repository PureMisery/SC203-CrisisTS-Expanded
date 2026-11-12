import os
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

os.makedirs('crisists/figures', exist_ok=True)
ARTIFACT_DIR = r'C:\Users\lqngu\.gemini\antigravity\brain\980ccb70-dd0a-4485-94e0-7565e1b061b5'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

with open('crisists/verified_5fold_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

with open('crisists/expanded_benchmark_results.json', 'r', encoding='utf-8') as f:
    exp_results = json.load(f)

with open('crisists/loeo_improved_results.json', 'r', encoding='utf-8') as f:
    loeo_results = json.load(f)

df = pd.read_csv('data/CrisisTS_Expanded_19Events.csv', sep='\t')


def parse_window(w):
    vals = list(map(float, str(w).replace('[', ' ').replace(']', ' ').split()))
    return np.array(vals).reshape(5, 11)


window_data = np.stack(df['Window'].apply(parse_window).values)  # (N, 5, 11)
X_weather = window_data.reshape(len(df), -1)  # (N, 55)
y_urgent = pd.factorize(df['label_urgent'], sort=True)[0]

# ============================================================
# FIGURE 1: 5-fold comparison bar chart with scatter overlay
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
tasks = ['label_urgent', 'label_humanitarian']
task_titles = ['label_urgent (3-class)', 'label_humanitarian (8-class)']
models = list(results[tasks[0]].keys())
colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
bar_width = 0.55
x_pos = np.arange(len(models))

for ax_idx, (task, title) in enumerate(zip(tasks, task_titles)):
    ax = axes[ax_idx]
    for m_idx, model in enumerate(models):
        fold_scores = results[task][model]
        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores, ddof=1)
        ax.bar(m_idx, mean_f1, bar_width, yerr=std_f1, capsize=4,
               color=colors[m_idx], edgecolor='white', linewidth=0.8,
               label=model if ax_idx == 0 else "")
        x_scatter = np.full(5, m_idx) + np.linspace(-0.12, 0.12, 5)
        ax.scatter(x_scatter, fold_scores, color='black', s=18, alpha=0.65, zorder=5, linewidths=0)
        ax.text(m_idx, mean_f1 + std_f1 + 0.02, f'{mean_f1:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center', fontsize=8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Macro-F1' if ax_idx == 0 else '')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('5-Fold Cross-Validation Macro-F1 Comparison (Original CrisisTS N=6,923)', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
path1 = 'crisists/figures/fig1_5fold_comparison.png'
fig.savefig(path1, dpi=300)
shutil.copy(path1, os.path.join(ARTIFACT_DIR, 'fig1_5fold_comparison.png'))
print(f'Saved: {path1}')
plt.close(fig)

# ============================================================
# FIGURE 2: Fold stability line plot (Expanded N=17,628)
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5))
exp_models = [
    ('B3_TS_RandomForest', 'Weather TS Only (RandomForest)'),
    ('B5_Text_LogReg', 'Text Only (Word+Char TF-IDF + LogReg)'),
    ('M1_Early_Feature_Fusion', 'Proposed M1: Early Feature Fusion'),
    ('M2_Stacking_Meta_Fusion', 'Proposed M2: Calibrated OOF Stacking')
]
markers = ['o', 's', '^', 'D']
linestyles = ['--', '-.', '-', '-']
line_colors = ['#e67e22', '#2980b9', '#27ae60', '#8e44ad']

for idx, (m_key, m_label) in enumerate(exp_models):
    fold_scores = exp_results['label_urgent']['summary'][m_key]['fold_macro_f1s']
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores, ddof=1)
    folds = np.arange(1, 6)
    ax.plot(folds, fold_scores, marker=markers[idx], linestyle=linestyles[idx], color=line_colors[idx],
            linewidth=2.2, markersize=7, label=f'{m_label} (μ={mean_f1:.4f}, σ={std_f1:.4f})')

ax.set_xlabel('Stratified Cross-Validation Fold (k = 1..5)')
ax.set_ylabel('Macro-F1 (label_urgent)')
ax.set_title('Fold-wise Stability Across All 19 Global Disasters (Expanded CrisisTS-19, N=17,628)', fontweight='bold')
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(0.70, 0.98)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.92)
fig.tight_layout()
path2 = 'crisists/figures/fig2_fold_stability.png'
fig.savefig(path2, dpi=300)
shutil.copy(path2, os.path.join(ARTIFACT_DIR, 'fig2_fold_stability.png'))
print(f'Saved: {path2}')
plt.close(fig)

# ============================================================
# FIGURE 3: Weather channel importance (Exact 11 Linker_Eng.py Channels)
# ============================================================
rf = RandomForestClassifier(n_estimators=200, max_depth=22, class_weight='balanced',
                            random_state=42, n_jobs=1)
rf.fit(X_weather, y_urgent)
importances = rf.feature_importances_.reshape(5, 11)  # (5 days, 11 channels)

# Exact 11 features from Linker_Eng.py lines 86-88
channel_names = [
    'Mean Wind Speed', 'Precipitation', 'Snowfall', 'Snow Depth',
    'Max Air Temp', 'Min Air Temp', 'Mean Air Temp',
    'Wind Dir (2-min)', 'Wind Dir (5-sec)', 'Max Wind (2-min)', 'Peak Gust (5-sec)'
]
day_labels = ['Day -4', 'Day -3', 'Day -2', 'Day -1', 'Day 0 (Event)']

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

ax = axes[0]
im = ax.imshow(importances, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xticks(np.arange(11))
ax.set_xticklabels(channel_names, rotation=40, ha='right')
ax.set_yticks(np.arange(5))
ax.set_yticklabels(day_labels)
ax.set_title('Feature Importance: 5-Day Window × 11 Weather Channels', fontweight='bold')
ax.set_xlabel('Meteorological Channel (Linker_Eng.py Exact Schema)')
ax.set_ylabel('Lag Day Relative to Tweet')
for i in range(5):
    for j in range(11):
        ax.text(j, i, f'{importances[i, j]:.3f}', ha='center', va='center',
                fontsize=7.5, color='black' if importances[i, j] < importances.max()*0.6 else 'white')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Gini Importance Weight')

ax = axes[1]
channel_totals = importances.sum(axis=0)
sorted_idx = np.argsort(channel_totals)[::-1]
y_pos = np.arange(11)
ax.barh(y_pos, channel_totals[sorted_idx], color=plt.cm.YlOrRd(0.65), edgecolor='white', height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([channel_names[i] for i in sorted_idx])
ax.invert_yaxis()
ax.set_xlabel('Cumulative 5-Day Importance')
ax.set_title('Total Importance per Physical Weather Channel (N=17,628)', fontweight='bold')
ax.set_xlim(0, max(channel_totals) * 1.18)
ax.grid(axis='x', alpha=0.3)
for i, idx in enumerate(sorted_idx):
    ax.text(channel_totals[idx] + 0.002, i, f'{channel_totals[idx]:.4f}', va='center', fontsize=9, fontweight='bold')

fig.suptitle('Physical Weather Channel Importance for Crisis Urgency (Expanded CrisisTS-19, N=17,628)', fontsize=13.5, fontweight='bold', y=1.02)
fig.tight_layout()
path3 = 'crisists/figures/fig3_weather_channel_importance.png'
fig.savefig(path3, dpi=300)
shutil.copy(path3, os.path.join(ARTIFACT_DIR, 'fig3_weather_channel_importance.png'))
print(f'Saved: {path3}')
plt.close(fig)

# ============================================================
# FIGURE 4: Expanded 4-Task Benchmark (N=17,628) & LOEO-CV Generalization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

# Panel A: 4 Tasks on Expanded N=17,628
ax = axes[0]
t_keys = ['label_urgent', 'label_humanitarian', 'label_useful', 'label_sudden']
t_labels = ['label_urgent\n(3-class)', 'label_humanitarian\n(8-class)', 'label_useful\n(2-class)', 'label_sudden\n(3-class)']
x = np.arange(len(t_keys))
w = 0.25

txt_vals = [exp_results[t]['summary']['B5_Text_LogReg']['macro_f1_mean'] for t in t_keys]
m1_vals = [exp_results[t]['summary']['M1_Early_Feature_Fusion']['macro_f1_mean'] for t in t_keys]
m2_vals = [exp_results[t]['summary']['M2_Stacking_Meta_Fusion']['macro_f1_mean'] for t in t_keys]

b1 = ax.bar(x - w, txt_vals, w, label='B5: Text-Only (LogReg)', color='#3498db', edgecolor='white')
b2 = ax.bar(x, m1_vals, w, label='M1: Multimodal Early Fusion', color='#2ecc71', edgecolor='white')
b3 = ax.bar(x + w, m2_vals, w, label='M2: Multimodal OOF Stacking', color='#9b59b6', edgecolor='white')

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(t_labels)
ax.set_ylim(0.65, 1.03)
ax.set_ylabel('5-Fold Macro-F1')
ax.set_title('A. Expanded CrisisTS-19 4-Task Benchmark (N=17,628)', fontweight='bold')
ax.legend(loc='lower right', frameon=True)
ax.grid(axis='y', alpha=0.3)

# Panel B: 4-Fold LOEO-CV (Leave-One-Crisis-Type-Out on Expanded N=17,628)
ax = axes[1]
ctypes = ['Hurricane', 'Earthquake', 'Wildfire', 'Flood']
x_c = np.arange(len(ctypes))
loeo_exp = exp_results['loeo_cv_expanded']
naive_loeo = [loeo_exp[c]['Naive_Linear_MM'] for c in ctypes]
txt_loeo = [loeo_exp[c]['Text_Only'] for c in ctypes]
imp_loeo = [loeo_exp[c]['Improved_Invariant_MM'] for c in ctypes]

c1 = ax.bar(x_c - w, naive_loeo, w, label='Naive Signed MM (Old)', color='#e74c3c', edgecolor='white')
c2 = ax.bar(x_c, txt_loeo, w, label='Unimodal Text-Only', color='#3498db', edgecolor='white')
c3 = ax.bar(x_c + w, imp_loeo, w, label='Proposed Sign-Invariant + Entropy Gate MM', color='#27ae60', edgecolor='white')

for bars in [c1, c2, c3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x_c)
ax.set_xticklabels([f'Held-Out\n{c}' for c in ctypes])
ax.set_ylim(0.0, 1.05)
ax.set_ylabel('Zero-Shot LOEO Macro-F1')
ax.set_title('B. Leave-One-Crisis-Type-Out (LOEO-CV) Generalization', fontweight='bold')
ax.legend(loc='lower right', frameon=True)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Global 19-Event Scaling (N=17,628) & Cross-Disaster Generalization', fontsize=13.5, fontweight='bold', y=1.02)
fig.tight_layout()
path4 = 'crisists/figures/fig4_expanded_19events_and_loeo.png'
fig.savefig(path4, dpi=300)
shutil.copy(path4, os.path.join(ARTIFACT_DIR, 'fig4_expanded_19events_and_loeo.png'))
print(f'Saved: {path4}')
plt.close(fig)
