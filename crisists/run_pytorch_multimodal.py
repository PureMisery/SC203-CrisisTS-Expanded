import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_window(w):
    vals = [float(x) for x in str(w).replace('[', ' ').replace(']', ' ').split()]
    arr = np.array(vals, dtype=np.float32)
    if arr.size != 55:
        arr = np.resize(arr, 55)
    return arr.reshape(5, 11)

class TS_1DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(11, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TS_BiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)

class Text_DeepMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Multimodal_BiLSTM_LateFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.lstm = nn.LSTM(input_size=11, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.ts_proj = nn.Linear(128, 64)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, text, ts):
        t = self.text_proj(text)
        _, (h_n, _) = self.lstm(ts)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.ts_proj(h)
        fused = torch.cat([t, h], dim=1)
        return self.classifier(fused)

class Multimodal_CrossAttention_Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_to_ctx = nn.Linear(128, 4 * 64)
        self.ts_lstm = nn.LSTM(input_size=11, hidden_size=64, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.ts_proj = nn.Linear(128, 64)
        self.attn_w2t = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.attn_t2w = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, text, ts):
        B = text.size(0)
        ctx = self.text_to_ctx(text).view(B, 4, 64)
        ts_out, _ = self.ts_lstm(ts)
        ts_proj = self.ts_proj(ts_out)
        w2t, _ = self.attn_w2t(ts_proj, ctx, ctx)
        w2t = self.ln1(w2t + ts_proj)
        t2w, _ = self.attn_t2w(ctx, w2t, w2t)
        t2w = self.ln2(t2w + ctx)
        pooled_w = w2t.mean(dim=1)
        pooled_t = t2w.mean(dim=1)
        fused = torch.cat([pooled_w, pooled_t], dim=1)
        return self.classifier(fused)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for *xb, yb in loader:
        xb = [x.to(device) for x in xb] if isinstance(xb, (list, tuple)) else xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(*xb) if isinstance(xb, (list, tuple)) else model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for *xb, yb in loader:
            xb = [x.to(device) for x in xb] if isinstance(xb, (list, tuple)) else xb.to(device)
            yb = yb.to(device)
            logits = model(*xb) if isinstance(xb, (list, tuple)) else model(xb)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return f1_score(trues, preds, average='macro')

def get_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def run_cv_for_label(df, label_col, num_classes, label_name):
    print(f"\n{'='*60}")
    print(f"Label: {label_name} ({num_classes} classes)")
    print(f"{'='*60}")
    
    y = pd.factorize(df[label_col], sort=True)[0]
    weather = np.stack(df['Window'].apply(parse_window).values)
    texts = df['Text'].fillna('').values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_config = {
        'TS_1DCNN': (TS_1DCNN, ['weather']),
        'TS_BiLSTM': (TS_BiLSTM, ['weather']),
        'Text_DeepMLP': (Text_DeepMLP, ['text']),
        'Multimodal_BiLSTM_LateFusion': (Multimodal_BiLSTM_LateFusion, ['text', 'weather']),
        'Multimodal_CrossAttention_Net': (Multimodal_CrossAttention_Net, ['text', 'weather']),
    }
    
    results = {name: [] for name in models_config}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(weather, y)):
        print(f"\n  Fold {fold+1}/5")
        
        y_train, y_test = y[train_idx], y[test_idx]
        weather_train, weather_test = weather[train_idx], weather[test_idx]
        text_train, text_test = texts[train_idx], texts[test_idx]
        
        scaler = StandardScaler()
        weather_train_flat = weather_train.reshape(-1, 11)
        weather_test_flat = weather_test.reshape(-1, 11)
        weather_train_scaled = scaler.fit_transform(weather_train_flat).reshape(-1, 5, 11)
        weather_test_scaled = scaler.transform(weather_test_flat).reshape(-1, 5, 11)
        
        tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1,2), sublinear_tf=True)
        X_train_tfidf = tfidf.fit_transform(text_train)
        X_test_tfidf = tfidf.transform(text_test)
        
        svd = TruncatedSVD(n_components=128, random_state=42)
        X_train_svd = svd.fit_transform(X_train_tfidf)
        X_test_svd = svd.transform(X_test_tfidf)
        
        scaler_text = StandardScaler()
        X_train_text = scaler_text.fit_transform(X_train_svd).astype(np.float32)
        X_test_text = scaler_text.transform(X_test_svd).astype(np.float32)
        
        weather_train_t = torch.tensor(weather_train_scaled, dtype=torch.float32)
        weather_test_t = torch.tensor(weather_test_scaled, dtype=torch.float32)
        text_train_t = torch.tensor(X_train_text, dtype=torch.float32)
        text_test_t = torch.tensor(X_test_text, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        
        class_weights = get_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for name, (ModelClass, inputs) in models_config.items():
            model = ModelClass(num_classes).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            
            if 'text' in inputs and 'weather' in inputs:
                train_ds = TensorDataset(text_train_t, weather_train_t, y_train_t)
                test_ds = TensorDataset(text_test_t, weather_test_t, y_test_t)
            elif 'text' in inputs:
                train_ds = TensorDataset(text_train_t, y_train_t)
                test_ds = TensorDataset(text_test_t, y_test_t)
            else:
                train_ds = TensorDataset(weather_train_t, y_train_t)
                test_ds = TensorDataset(weather_test_t, y_test_t)
            
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            
            for epoch in range(25):
                train_epoch(model, train_loader, criterion, optimizer)
            
            f1 = evaluate(model, test_loader)
            results[name].append(f1)
            print(f"    {name}: Macro-F1 = {f1:.4f}")
    
    return results

def main():
    df = pd.read_csv('data/CrisisTS_Eng.csv', sep='\t')
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    all_results = {}
    
    for label_col, num_classes, label_name in [
        ('label_urgent', 3, 'Urgent'),
        ('label_humanitarian', 8, 'Humanitarian')
    ]:
        results = run_cv_for_label(df, label_col, num_classes, label_name)
        all_results[label_name] = results
        
        print(f"\n  Summary for {label_name}:")
        text_f1s = results['Text_DeepMLP']
        for name, f1s in results.items():
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            if name != 'Text_DeepMLP':
                t_stat, p_val = ttest_rel(f1s, text_f1s)
                print(f"    {name}: {mean_f1:.4f} ± {std_f1:.4f}  (t={t_stat:.3f}, p={p_val:.4f} vs Text_DeepMLP)")
            else:
                print(f"    {name}: {mean_f1:.4f} ± {std_f1:.4f}  (reference)")
    
    os.makedirs('crisists', exist_ok=True)
    with open('crisists/pytorch_multimodal_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to crisists/pytorch_multimodal_results.json")

if __name__ == '__main__':
    main()