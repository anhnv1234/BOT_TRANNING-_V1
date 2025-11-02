# File: context_agent.py (v15.1 - Trí Tuệ Cảm Xúc)
# Agent tối cao, phiên bản hoàn chỉnh, xử lý thông minh các trường hợp thiếu tín hiệu.

import pandas as pd
import pandas_ta as ta
import yaml
from pathlib import Path
import logging
import numpy as np
import requests
from scipy.signal import find_peaks
from datetime import timezone
import pylunar
from tqdm import tqdm

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContextAgent:
    def __init__(self, config_path='config_resample.yaml'):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.params = self.config['context_agent']
            self.data_path = Path(self.params['h1_data_path'])
            logging.info("Đã khởi tạo Context Agent với các tham số: %s", self.params)
        except (FileNotFoundError, KeyError) as e:
            logging.error(f"Lỗi nghiêm trọng khi khởi tạo: {e}. Vui lòng kiểm tra file config '{config_path}'.")
            self.params = None

    def _fetch_historical_ohlc(self, limit):
        logging.info(f"Đang lấy {limit} nến H1 lịch sử (OHLC) từ Binance...")
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': limit}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df = df[['timestamp', 'open', 'high', 'low', 'close']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            return df.set_index('timestamp')
        except Exception as e:
            logging.error(f"Lỗi khi lấy dữ liệu khởi động nóng: {e}")
            return None

    def _hot_start_data(self, df):
        required_rows = self.params['trend_filter_sma_period']
        if len(df) < required_rows:
            logging.warning(f"Dữ liệu không đủ ({len(df)}/{required_rows}). Bắt đầu khởi động nóng.")
            num_to_fetch = 240
            historical_data = self._fetch_historical_ohlc(limit=num_to_fetch)
            if historical_data is not None:
                df_combined = pd.concat([historical_data, df], sort=True)
                for col in ['open', 'high', 'low', 'close']:
                    df_combined[col] = df_combined[col].ffill()
                logging.info("Khởi động nóng trong bộ nhớ thành công.")
                return df_combined
        return df
        
    def _get_astrological_sign(self, date_time):
        try:
            utc_dt = date_time.astimezone(timezone.utc)
            hanoi_lat, hanoi_lon = (21.028511, 105.804817)
            moon = pylunar.MoonInfo(hanoi_lat, hanoi_lon)
            moon.update(utc_dt)
            phase_name = moon.phase_name().upper()
            if "NEW MOON" in phase_name: return 1.0
            elif "FULL MOON" in phase_name: return -1.0
            elif "FIRST QUARTER" in phase_name: return 0.5
            elif "LAST QUARTER" in phase_name: return -0.5
            else: return 0.0
        except Exception as e:
            logging.warning(f"Không thể luận giải thiên văn: {e}")
            return 0.0

    def _calculate_indicators(self, df):
        df.ta.ema(close=df['close'], length=self.params['fast_ema_period'], append=True)
        df.ta.ema(close=df['close'], length=self.params['slow_ema_period'], append=True)
        df.ta.sma(close=df['close'], length=self.params['trend_filter_sma_period'], append=True)
        df.ta.rsi(close=df['close'], length=self.params['rsi_period'], append=True)
        df.ta.atr(high=df['high'], low=df['low'], close=df['close'], length=self.params['atr_period'], append=True)
        df['oi_momentum'] = df['open_interest'].diff()
        df['delta_momentum'] = df['delta'].diff()
        df.dropna(subset=[f"SMA_{self.params['trend_filter_sma_period']}"], inplace=True)
        return df

    def _detect_divergence(self, df):
        price = df['close']
        rsi = df[f"RSI_{self.params['rsi_period']}"]
        peak_distance = self.params['divergence_peak_distance']
        price_peaks, _ = find_peaks(price, distance=peak_distance, prominence=price.std()*0.5)
        rsi_peaks, _ = find_peaks(rsi, distance=peak_distance)
        price_troughs, _ = find_peaks(-price, distance=peak_distance, prominence=price.std()*0.5)
        rsi_troughs, _ = find_peaks(-rsi, distance=peak_distance)
        divergence_score = 0.0
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if price.iloc[price_peaks[-1]] > price.iloc[price_peaks[-2]] and rsi.iloc[rsi_peaks[-1]] < rsi.iloc[rsi_peaks[-2]]:
                price_change_pct = (price.iloc[price_peaks[-1]] - price.iloc[price_peaks[-2]]) / (price.iloc[price_peaks[-2]] + 1e-9)
                rsi_change_pct = (rsi.iloc[rsi_peaks[-2]] - rsi.iloc[rsi_peaks[-1]]) / (rsi.iloc[rsi_peaks[-2]] + 1e-9)
                divergence_score = -1 * price_change_pct * rsi_change_pct
        elif len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if price.iloc[price_troughs[-1]] < price.iloc[price_troughs[-2]] and rsi.iloc[rsi_troughs[-1]] > rsi.iloc[rsi_troughs[-2]]:
                price_change_pct = (price.iloc[price_troughs[-2]] - price.iloc[price_troughs[-1]]) / (price.iloc[price_troughs[-2]] + 1e-9)
                rsi_change_pct = (rsi.iloc[rsi_troughs[-1]] - rsi.iloc[rsi_troughs[-2]]) / (rsi.iloc[rsi_troughs[-2]] + 1e-9)
                divergence_score = price_change_pct * rsi_change_pct
        return divergence_score

    def _calculate_single_fan(self, df_slice, pivot_type):
        default_result = {"line_1x1_price": np.nan, "line_1x2_price": np.nan, "line_1x3_price": np.nan, "line_2x1_price": np.nan, "price_vs_1x3": np.nan, "position_score": 0, "continuous_score": 0.0}
        atr_col_name = f"ATRr_{self.params['atr_period']}"
        if pivot_type == 'low':
            pivots, _ = find_peaks(-df_slice['low'], distance=5, prominence=df_slice[atr_col_name].mean()*0.5)
            if len(pivots) == 0: pivots = [df_slice['low'].argmin()]
            pivot_iloc = pivots[-1]
            pivot_idx = df_slice.index[pivot_iloc]
            pivot_price = df_slice.iloc[pivot_iloc]['low']
            fan_direction = 1
        else:
            pivots, _ = find_peaks(df_slice['high'], distance=5, prominence=df_slice[atr_col_name].mean()*0.5)
            if len(pivots) == 0: pivots = [df_slice['high'].argmax()]
            pivot_iloc = pivots[-1]
            pivot_idx = df_slice.index[pivot_iloc]
            pivot_price = df_slice.iloc[pivot_iloc]['high']
            fan_direction = -1
        pivot_atr = df_slice.loc[pivot_idx, atr_col_name]
        if pd.isna(pivot_atr) or pivot_atr == 0: pivot_atr = 1e-9
        time_delta = len(df_slice) - 1 - pivot_iloc
        latest_price = df_slice['close'].iloc[-1]
        price_change_unit = pivot_atr
        gann_1x1 = pivot_price + fan_direction * time_delta * price_change_unit
        gann_1x2 = pivot_price + fan_direction * time_delta * (price_change_unit * 2)
        gann_1x3 = pivot_price + fan_direction * time_delta * (price_change_unit * 3)
        gann_2x1 = pivot_price + fan_direction * time_delta * (price_change_unit * 0.5)
        gann_position_score, gann_continuous_score = 0, 0.0
        epsilon = 1e-9
        if fan_direction == 1:
            if latest_price > gann_1x2: gann_position_score, gann_continuous_score = 3, 2.0 + (latest_price - gann_1x2) / (pivot_atr + epsilon)
            elif latest_price > gann_1x1: gann_position_score, gann_continuous_score = 2, 1.0 + (latest_price - gann_1x1) / (abs(gann_1x2 - gann_1x1) + epsilon)
            elif latest_price > gann_2x1: gann_position_score, gann_continuous_score = 1, (latest_price - gann_2x1) / (abs(gann_1x1 - gann_2x1) + epsilon)
            else: gann_position_score, gann_continuous_score = 0, (latest_price - gann_2x1) / (pivot_atr + epsilon)
        elif fan_direction == -1:
            if latest_price < gann_1x2: gann_position_score, gann_continuous_score = -3, -2.0 + (latest_price - gann_1x2) / (pivot_atr + epsilon)
            elif latest_price < gann_1x1: gann_position_score, gann_continuous_score = -2, -1.0 + (latest_price - gann_1x1) / (abs(gann_1x2 - gann_1x1) + epsilon)
            elif latest_price < gann_2x1: gann_position_score, gann_continuous_score = -1, (latest_price - gann_2x1) / (abs(gann_1x1 - gann_2x1) + epsilon)
            else: gann_position_score, gann_continuous_score = 0, (latest_price - gann_2x1) / (pivot_atr + epsilon)
        default_result.update({"line_1x1_price": gann_1x1, "line_1x2_price": gann_1x2, "line_1x3_price": gann_1x3, "line_2x1_price": gann_2x1, "price_vs_1x3": (latest_price - gann_1x3) / (latest_price + epsilon), "position_score": gann_position_score, "continuous_score": np.clip(gann_continuous_score, -5, 5)})
        return default_result

    def _calculate_dual_gann_fan_features(self, df):
        lookback = self.params['gann_pivot_lookback']
        df_slice = df.tail(lookback).copy()
        up_fan_features = self._calculate_single_fan(df_slice, 'low')
        down_fan_features = self._calculate_single_fan(df_slice, 'high')
        combined_features = {}
        for key, value in up_fan_features.items():
            combined_features[f'gann_up_{key}'] = value
        for key, value in down_fan_features.items():
            combined_features[f'gann_down_{key}'] = value
        return combined_features
        
    def _calculate_master_confidence(self, assessment):
        weights = {'trend': 0.50, 'gann': 0.30, 'divergence': 0.15, 'moon': 0.05}
        score1 = assessment.get('trend_confirmation_score', 0) / 2.0
        score4 = assessment.get('moon_phase_score', 0.0)
        gann_up_price = assessment.get('gann_up_line_1x1_price')
        gann_is_valid = gann_up_price is not None and pd.notna(gann_up_price)
        if gann_is_valid:
            gann_up = assessment.get('gann_up_continuous_score', 0)
            gann_down = assessment.get('gann_down_continuous_score', 0)
            gann_score_normalized = np.clip((gann_up + gann_down) / 2, -3, 3) / 3.0
            score2 = gann_score_normalized
            core_scores = [score1, score2]
            final_trend_weight, final_gann_weight = weights['trend'], weights['gann']
        else:
            score2 = 0
            core_scores = [score1]
            final_trend_weight, final_gann_weight = weights['trend'] + weights['gann'], 0
        disagreement = np.std(core_scores)
        consistency = 1 - min(disagreement, 1.0)
        assessment['signal_consistency'] = consistency
        divergence_score = assessment.get('divergence_score', 0.0)
        divergence_impact = 0
        if assessment.get('trend_direction', 0) > 0 and divergence_score < 0:
            divergence_impact = np.clip(divergence_score / -0.01, -1, 0)
        elif assessment.get('trend_direction', 0) < 0 and divergence_score > 0:
            divergence_impact = np.clip(divergence_score / 0.01, 0, 1)
        main_trend_confidence = (score1 * final_trend_weight) + (score2 * final_gann_weight) + (score4 * weights['moon'])
        final_confidence = main_trend_confidence + (divergence_impact * weights['divergence'])
        final_confidence *= consistency
        return np.clip(final_confidence, -1, 1)

    def get_market_assessment(self):
        if not self.params: return {}
        steps = ["Tải Dữ Liệu", "Khởi Động Nóng", "Tính Chỉ Báo", "Phân Tích Nâng Cao", "Tổng Hợp Báo Cáo"]
        with tqdm(total=len(steps), desc="Thái Thượng Hoàng Luận Giải") as pbar:
            try:
                pbar.set_description(steps[0]); pbar.refresh()
                df_raw = pd.read_parquet(self.data_path)
                df_raw.columns = [c.replace('_sum', '').replace('_last', '').replace('_max','') for c in df_raw.columns]
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                df_raw.set_index('timestamp', inplace=True)
                pbar.update(1)
            except FileNotFoundError:
                logging.error(f"File dữ liệu '{self.data_path}' không tồn tại.")
                return {}

            pbar.set_description(steps[1]); pbar.refresh()
            df_primed = self._hot_start_data(df_raw)
            if not df_primed.index.tz:
                df_primed = df_primed.tz_localize('UTC')
            pbar.update(1)

            pbar.set_description(steps[2]); pbar.refresh()
            df_with_indicators = self._calculate_indicators(df_primed)
            if df_with_indicators is None or df_with_indicators.empty:
                logging.warning("Không có đủ dữ liệu để tính toán chỉ báo sau khi lọc.")
                return {}
            pbar.update(1)

            pbar.set_description(steps[3]); pbar.refresh()
            latest_data = df_with_indicators.iloc[-1]
            divergence_lookback_df = df_with_indicators.tail(self.params['divergence_lookback'])
            moon_score = self._get_astrological_sign(latest_data.name)
            divergence = self._detect_divergence(divergence_lookback_df)
            gann_features = self._calculate_dual_gann_fan_features(df_with_indicators)
            pbar.update(1)

            pbar.set_description(steps[4]); pbar.refresh()
            price = latest_data['close']
            fast_ema = latest_data[f"EMA_{self.params['fast_ema_period']}"]
            slow_ema = latest_data[f"EMA_{self.params['slow_ema_period']}"]
            trend_filter = latest_data[f"SMA_{self.params['trend_filter_sma_period']}"]
            trend_strength = abs(fast_ema - slow_ema) / (price + 1e-9)
            trend_direction = 0
            if trend_strength > self.params['trend_strength_threshold']:
                if price > trend_filter and fast_ema > slow_ema:
                    trend_direction = 1
                elif price < trend_filter and fast_ema < slow_ema:
                    trend_direction = -1
            
            trend_confirmation_score = 0
            oi_momentum = latest_data['oi_momentum'] if 'oi_momentum' in latest_data and pd.notna(latest_data['oi_momentum']) else 0
            if trend_direction == 1: trend_confirmation_score = 2 if oi_momentum > 0 else 1
            elif trend_direction == -1: trend_confirmation_score = -2 if oi_momentum > 0 else -1
            
            assessment = {'trend_direction': trend_direction, 'trend_strength': trend_strength, 'price_vs_filter': (price - trend_filter) / (trend_filter + 1e-9), 'oi_momentum': oi_momentum, 'delta_momentum': latest_data['delta_momentum'] if pd.notna(latest_data['delta_momentum']) else 0, 'trend_confirmation_score': trend_confirmation_score, 'divergence_score': divergence, 'moon_phase_score': moon_score}
            assessment.update(gann_features)
            
            master_confidence = self._calculate_master_confidence(assessment)
            assessment['master_confidence_score'] = master_confidence
            pbar.update(1)
            
        logging.info("Đã hoàn tất bản báo cáo tình hình thị trường.")
        return assessment

if __name__ == "__main__":
    print("\n" + "="*60)
    print("====== KHỞI ĐỘNG THÁI THƯỢNG HOÀNG (CONTEXT AGENT) v15.1 ======")
    print("="*60 + "\n")

    thai_thuong_hoang = ContextAgent()
    assessment = thai_thuong_hoang.get_market_assessment()

    print("\n" + "-"*60)
    print(" BÁO CÁO TÌNH HÌNH THỊ TRƯỜNG TỪ THÁI THƯỢNG HOÀNG")
    print("-"*60)
    
    if not assessment:
        print("Không có đủ dữ liệu để tạo báo cáo.")
    else:
        for key, value in assessment.items():
            if isinstance(value, float):
                print(f"- {key:<28}: {value:.6f}")
            else:
                print(f"- {key:<28}: {value}")
    
    print("-"*60)