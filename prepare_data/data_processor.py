# File: data_processor.py (v2.4 - Thêm Thanh Tiến Trình)
import pandas as pd
import yaml
import logging
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm # <<< BỔ SUNG THƯ VIỆN TIẾN TRÌNH

# --- Thiết lập Logging đẹp đẽ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Hàm định nghĩa "Công Thức Nấu Ăn" (Quy tắc tổng hợp dữ liệu) ---
def get_aggregation_rules(columns):
    rules = {}
    # Tính OHLC từ mid_price
    rules['mid_price'] = ['first', 'max', 'min', 'last'] 
    
    # Các cột giá khác và các chỉ số trạng thái lấy giá trị cuối cùng
    price_cols = [col for col in columns if 'mark_price' in col or 'rate' in col or 'interest' in col or 'spread' in col]
    for col in price_cols: rules[col] = 'last'
    
    # Các cột khối lượng và delta thì cộng dồn
    volume_cols = [col for col in columns if 'volume' in col or 'delta' in col]
    for col in volume_cols: rules[col] = 'sum'
    
    # Cột phụ để tính VWAP
    if 'pxv' in columns:
        rules['pxv'] = 'sum'
        
    # Các cột size của liquidity thì lấy cả giá trị cuối và lớn nhất
    size_cols = [col for col in columns if 'size' in col]
    for col in size_cols: rules[col] = ['last', 'max']
    
    # Các cột còn lại của liquidity (ratio, imbalance, giá sig) lấy giá trị cuối
    other_cols = [col for col in columns if 'ratio' in col or 'imbalance' in col or 'sig' in col]
    for col in other_cols: rules[col] = 'last'
    
    return rules

# --- Hàm Chính: "Tổng Quản Nhà Bếp" ---
def process_and_resample(config):
    input_path = Path(config['input_file_path'])
    output_folder = Path(config['output_folder'])
    output_files = config['output_files']
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Thư mục đầu ra '{output_folder}' đã sẵn sàng.")
    try:
        logger.info(f"Bắt đầu đọc dữ liệu thô từ '{input_path}'...")
        df = pd.read_parquet(input_path)
        logger.info(f"Đọc thành công! Dữ liệu có {len(df)} dòng.")
    except FileNotFoundError:
        logger.error(f"LỖI: Không tìm thấy file dữ liệu đầu vào tại '{input_path}'.")
        return

    logger.info("Bắt đầu sơ chế...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['total_volume_1s'] = df.get('buy_volume_1s', 0) + df.get('sell_volume_1s', 0)
    df['pxv'] = df.get('mid_price', 0) * df['total_volume_1s']
    df.set_index('timestamp', inplace=True)

    logger.info("Lấp đầy các giây bị thiếu...")
    df_filled = df.resample('1s').ffill()
    
    aggregation_rules = get_aggregation_rules(df_filled.columns)
    
    timeframes = {
        "Cơm Rang Tốc Độ (3 Giây)": ("3s", output_folder / output_files['bookmap_3s']),
        "Phở Đặc Biệt (1 Phút)": ("1min", output_folder / output_files['council_1m']),
        "Đại Tiệc Hoàng Cung (1 Giờ)": ("1h", output_folder / output_files['emperor_1h']),
    }

    # <<< THÊM THANH TIẾN TRÌNH TẠI ĐÂY >>>
    for name, (freq, output_path) in tqdm(timeframes.items(), desc="Đang Chế Biến Các Khung Thời Gian"):
        logger.info(f"====== Bắt đầu chế biến món: '{name}' ======")
        resampled_df = df_filled.resample(freq).agg(aggregation_rules)
        resampled_df.dropna(how='all', inplace=True)
        
        resampled_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in resampled_df.columns.values]
        
        resampled_df.rename(columns={
            'mid_price_first': 'open', 'mid_price_max': 'high', 
            'mid_price_min': 'low', 'mid_price_last': 'close'
        }, inplace=True)

        if 'pxv_sum' in resampled_df.columns and 'total_volume_1s_sum' in resampled_df.columns:
            resampled_df['vwap'] = np.divide(
                resampled_df['pxv_sum'], resampled_df['total_volume_1s_sum'], 
                out=np.zeros_like(resampled_df['pxv_sum']), 
                where=resampled_df['total_volume_1s_sum']!=0
            )
            resampled_df.drop(columns=['pxv_sum'], inplace=True)
            
        resampled_df.reset_index(inplace=True)
        
        if not resampled_df.empty:
            logger.info(f"Đang lưu thành phẩm ra file '{output_path}'...")
            resampled_df.to_parquet(output_path, index=False)
            logger.info(f"Lưu thành công cho '{name}'!")

    logger.info("\n" + "="*50)
    logger.info("TOÀN BỘ QUÁ TRÌNH CHẾ BIẾN DỮ LIỆU ĐÃ HOÀN TẤT!")
    logger.info(f"Bệ hạ có thể tìm thấy các món ăn trong thư mục: '{output_folder}'")
    logger.info("="*50)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("====== NHÀ MÁY CHẾ BIẾN DỮ LIỆU GIAO DỊCH v2.4 ======")
    print("="*60 + "\n")
    try:
        with open('config_resample.yaml', 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
        process_and_resample(CONFIG)
    except FileNotFoundError:
        logger.error("LỖI: Không tìm thấy file 'config_resample.yaml'.")