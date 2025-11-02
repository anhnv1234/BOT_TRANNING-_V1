# File: inspect_datafile.py
# Dùng để kiểm tra cấu trúc của một file Parquet

import pandas as pd
from pathlib import Path

# Đường dẫn tới file dữ liệu cần kiểm tra
# Hãy đảm bảo đường dẫn này chính xác
file_to_inspect = Path("processed_data/data_1h_emperor.parquet")

print("="*50)
print(f"BẮT ĐẦU CHẨN ĐOÁN FILE: {file_to_inspect}")
print("="*50)

try:
    df = pd.read_parquet(file_to_inspect)
    
    print("\n[THÔNG TIN FILE]")
    df.info()
    
    print("\n[DANH SÁCH CÁC CỘT HIỆN CÓ]")
    print(list(df.columns))
    
    print("\n[5 DÒNG DỮ LIỆU ĐẦU TIÊN]")
    print(df.head())

except FileNotFoundError:
    print(f"\nLỖI: Không tìm thấy file tại '{file_to_inspect}'. Bệ hạ cần chạy data_processor.py trước.")
except Exception as e:
    print(f"\nLỖI KHÁC: {e}")

print("\n" + "="*50)
print("CHẨN ĐOÁN KẾT THÚC")
print("="*50)