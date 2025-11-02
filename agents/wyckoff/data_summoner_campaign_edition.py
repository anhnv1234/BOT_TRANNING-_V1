# File: optimizer_wyckoff.py
# PHIÊN BẢN HOÀNG KIM - V4 (Tuyển Tướng Toàn Diện - Nhiều Thông Số)

import pandas as pd
import optuna
from tqdm.autonotebook import tqdm
import numpy as np
import sys
import os
from wyckoff_agent import WyckoffAgent

# ======================================================================
# BỆ HẠ CHỈ CẦN CHỈNH Ở ĐÂY
# ======================================================================
DATA_FILE = 'data_1m_BTCUSDT_6_months.parquet'
N_TRIALS_PER_LEAGUE = 50 
FEE = 0.0004
INITIAL_CAPITAL = 10000.0
# ======================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)

def block_print():
    sys.stdout = open(os.devnull, 'w')
def enable_print():
    sys.stdout = sys.__stdout__

def run_backtest(df, fee, initial_capital):
    capital = initial_capital
    position = 0
    entry_price = 0
    wins, losses, total_trades, buy_trades, sell_trades = 0, 0, 0, 0, 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="    -> Đang mô phỏng", leave=False):
        if position == 1 and row['wyckoff_decision'] == 'SELL':
            pnl = (row['close'] - entry_price) / entry_price; capital *= (1 + pnl - fee)
            if pnl > 0: wins += 1
            else: losses += 1
            position = 0; total_trades += 1
        elif position == -1 and row['wyckoff_decision'] == 'BUY':
            pnl = (entry_price - row['close']) / entry_price; capital *= (1 + pnl - fee)
            if pnl > 0: wins += 1
            else: losses += 1
            position = 0; total_trades += 1
        
        if position == 0:
            if row['wyckoff_decision'] == 'BUY':
                position = 1; entry_price = row['close']; capital *= (1 - fee); buy_trades += 1
            elif row['wyckoff_decision'] == 'SELL':
                position = -1; entry_price = row['close']; capital *= (1 - fee); sell_trades += 1
    
    win_loss_ratio = wins / losses if losses > 0 else float(wins)
    return {
        "pnl_percent": ((capital - initial_capital) / initial_capital) * 100,
        "wins": wins, "losses": losses, "total_trades": total_trades,
        "buy_trades": buy_trades, "sell_trades": sell_trades, "win_loss_ratio": win_loss_ratio
    }

def objective(trial, data_df, metric_to_optimize):
    # === THÊM CÁC THÔNG SỐ MỚI VÀO QUÁ TRÌNH TUYỂN CHỌN ===
    params = {
        'lookback_period': trial.suggest_int('lookback_period', 20, 100, step=5),
        'volume_multiplier': trial.suggest_float('volume_multiplier', 1.5, 3.5),
        'test_volume_multiplier': trial.suggest_float('test_volume_multiplier', 0.5, 1.5),
        'atr_period': trial.suggest_int('atr_period', 10, 30),
        'decision_threshold': trial.suggest_float('decision_threshold', 2.0, 5.0),
        
        'event_thresholds': {
            'spring_close_pos': trial.suggest_float('spring_close_pos', 0.5, 0.8),
            'upthrust_close_pos': trial.suggest_float('upthrust_close_pos', 0.2, 0.5),
            'no_supply_demand_spread_window': trial.suggest_int('no_supply_demand_spread_window', 5, 20),
            'climax_sell_close_pos': trial.suggest_float('climax_sell_close_pos', 0.1, 0.35),
            'climax_buy_close_pos': trial.suggest_float('climax_buy_close_pos', 0.65, 0.9),
        },
        'fsm': {
            'reset_period': trial.suggest_int('reset_period', 15, 50),
            'exhaustion_cluster_size': trial.suggest_int('exhaustion_cluster_size', 2, 5),
        },
        'event_buffer': {
            'enabled': True, # Luôn bật để tìm độ điềm tĩnh
            'min_cluster_size': trial.suggest_int('min_cluster_size', 2, 5),
        }
    }
    config_dict = {'council_agents': {'wyckoff': params}}
    
    block_print()
    agent = WyckoffAgent(config_dict=config_dict)
    analyzed_df = agent.analyze(data_df)
    enable_print()

    if analyzed_df is None or len(analyzed_df[analyzed_df['wyckoff_decision'] != 'HOLD']) < 10:
        return -1000.0

    stats = run_backtest(analyzed_df, FEE, INITIAL_CAPITAL)
    
    print(f"\n[Vòng {trial.number}] PNL: {stats['pnl_percent']:.2f}% | "
          f"W/L: {stats['win_loss_ratio']:.2f} ({stats['wins']} Thắng / {stats['losses']} Thua) | "
          f"Tổng lệnh: {stats['total_trades']}")

    if metric_to_optimize == 'pnl':
        return stats['pnl_percent']
    else:
        return stats['win_loss_ratio']

def run_optimization():
    print("="*60)
    print("⚔️ MỞ CỬA ĐẤU TRƯỜNG TOÀN DIỆN - WYCKOFF VƯƠNG GIẢ ⚔️")
    print("="*60)
    
    try:
        df = pd.read_parquet(DATA_FILE)
        print(f"✅ Đã triệu hồi thành công sa bàn từ file: {DATA_FILE}")
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file '{DATA_FILE}'.")
        return

    print("\n" + "="*20 + " BẮT ĐẦU GIẢI ĐẤU 1: PNL VÔ ĐỊCH " + "="*20)
    study_pnl = optuna.create_study(direction='maximize')
    with tqdm(total=N_TRIALS_PER_LEAGUE, desc="[GIẢI PNL]") as pbar:
        study_pnl.optimize(lambda trial: objective(trial, df, 'pnl'), n_trials=N_TRIALS_PER_LEAGUE, 
                           callbacks=[lambda st, tr: pbar.update(1)])

    print("\n" + "="*15 + " BẮT ĐẦU GIẢI ĐẤU 2: THƯỜNG THẮNG TƯỚNG QUÂN " + "="*15)
    study_wl = optuna.create_study(direction='maximize')
    with tqdm(total=N_TRIALS_PER_LEAGUE, desc="[GIẢI W/L]") as pbar:
        study_wl.optimize(lambda trial: objective(trial, df, 'win_loss_ratio'), n_trials=N_TRIALS_PER_LEAGUE, 
                           callbacks=[lambda st, tr: pbar.update(1)])

    print("\n" + "="*60)
    print("🎉🎉 ĐẤU TRƯỜNG TOÀN DIỆN ĐÃ KẾT THÚC! 🎉🎉")
    
    print("\n" + "-"*25 + " 🏆 PNL VÔ ĐỊCH 🏆 " + "-"*25)
    print(f"Vị tướng kiếm nhiều tiền nhất đạt PNL: {study_pnl.best_value:.2f}%")
    print("    -> Bộ thông số hoàng kim:")
    for key, value in study_pnl.best_params.items():
        print(f"        - {key}: {value}")
        
    print("\n" + "-"*20 + " 🏅 THƯỜNG THẮNG TƯỚNG QUÂN 🏅 " + "-"*20)
    print(f"Vị tướng bách chiến bách thắng đạt Tỷ lệ W/L: {study_wl.best_value:.2f}")
    print("    -> Bộ thông số hoàng kim:")
    for key, value in study_wl.best_params.items():
        print(f"        - {key}: {value}")

if __name__ == '__main__':
    run_optimization()