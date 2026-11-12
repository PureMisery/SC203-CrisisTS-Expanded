"""
Script tải nhanh tập dữ liệu CrisisTS (English Multimodal Subset) từ Hugging Face
Phục vụ học phần SC203 - ĐHKHTN (VNUHCM-US)
"""
import os, sys
import requests
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

HF_URL = "https://huggingface.co/datasets/Unknees/CrisisTS/raw/main/Multi_modal_dataset/Multi_modal_ENG/CrisisTS_Eng.csv"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "CrisisTS_Eng.csv")

def download_crisists():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_CSV):
        print(f"[+] Tệp đã tồn tại: {OUT_CSV}")
    else:
        print(f"[*] Đang tải CrisisTS từ Hugging Face...")
        r = requests.get(HF_URL, stream=True)
        r.raise_for_status()
        with open(OUT_CSV, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print(f"[✓] Tải thành công về: {OUT_CSV}")

    # Kiểm tra nhanh dữ liệu
    df = pd.read_csv(OUT_CSV, sep="\t")
    print("\n=== THỐNG KÊ NHANH DỮ LIỆU CRISISTS ===")
    print(f"- Tổng số mẫu đa phương thức: {len(df):,}")
    print(f"- Các cột: {df.columns.tolist()}")
    print("\nPhân phối loại thiên tai (Crisis_Type):")
    print(df['Crisis_Type'].value_counts())
    print("\nPhân phối nhãn khẩn cấp (label_urgent):")
    print(df['label_urgent'].value_counts())
    print("\nPhân phối nhãn nhân đạo (label_humanitarian):")
    print(df['label_humanitarian'].value_counts())

if __name__ == "__main__":
    download_crisists()
