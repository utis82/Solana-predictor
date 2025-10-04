import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import math
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=512, nhead=16, num_layers=8, 
                 dim_feedforward=2048, dropout=0.15, num_outputs=3):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 128)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_outputs)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_model():
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    scaler = joblib.load('models/scaler.pkl')
    model = TransformerModel(
        input_size=metadata['input_size'],
        d_model=512, nhead=16, num_layers=8,
        dim_feedforward=2048, dropout=0.15, num_outputs=3
    ).to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()
    return model, scaler, metadata

def prepare_features(df):
    feature_cols = [
        'price', 'volume', 'open', 'high', 'low',
        'ma_7', 'ma_25', 'ma_99', 'rsi', 'macd', 'macd_signal',
        'volatility', 'price_change', 'price_change_1h', 'price_change_24h',
        'volume_change', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower'
    ]
    return df[feature_cols].values

def make_prediction(model, scaler, sequence):
    sequence_scaled = scaler.transform(sequence)
    sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(sequence_tensor)
    return prediction.cpu().numpy()[0]

def simulate_strategy(df, model, scaler, buy_threshold, sell_threshold, 
                     test_start='2024-09-01', sequence_length=72):
    """Simule une stratégie avec des seuils donnés"""
    
    df_test = df[df.index >= test_start].copy()
    
    capital = 100000
    position = 0
    entry_price = None
    trades = 0
    
    features = prepare_features(df)
    features = np.where(np.isinf(features), np.nan, features)
    mask = ~np.any(np.isnan(features), axis=1)
    features = features[mask]
    df_clean = df[mask].copy()
    
    test_start_idx = df_clean.index.searchsorted(df_test.index[0])
    
    for i in range(test_start_idx, len(features) - 1):
        if i < sequence_length:
            continue
        
        sequence = features[i-sequence_length:i]
        current_price = df_clean.iloc[i]['price']
        
        predictions = make_prediction(model, scaler, sequence)
        predicted_change = predictions[2]  # 24h
        
        # Achat
        if predicted_change > buy_threshold and position == 0 and capital > 0:
            amount = capital * 0.3
            fee = amount * 0.003
            position = (amount - fee) / current_price
            capital -= amount
            entry_price = current_price
            trades += 1
        
        # Vente
        elif predicted_change < sell_threshold and position > 0:
            revenue = position * current_price
            fee = revenue * 0.003
            capital += (revenue - fee)
            position = 0
            entry_price = None
            trades += 1
        
        # Stop loss
        elif position > 0 and entry_price:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= -0.12:
                revenue = position * current_price
                fee = revenue * 0.003
                capital += (revenue - fee)
                position = 0
                entry_price = None
    
    # Liquidation
    if position > 0:
        final_price = df_clean.iloc[-1]['price']
        revenue = position * final_price
        fee = revenue * 0.003
        capital += (revenue - fee)
    
    roi = ((capital / 100000) - 1) * 100
    
    return {
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'final_capital': capital,
        'roi': roi,
        'trades': trades
    }

def main():
    print("=" * 70)
    print("OPTIMISATION DES SEUILS")
    print("=" * 70)
    
    model, scaler, metadata = load_model()
    df = pd.read_csv('data/solana_prepared.csv', index_col='timestamp', parse_dates=True)
    
    print("\nTest de différents seuils...")
    
    # Grille de recherche
    buy_thresholds = [1.0, 1.5, 2.0, 2.5]
    sell_thresholds = [-1.0, -1.5, -2.0, -2.5]
    
    results = []
    
    total = len(buy_thresholds) * len(sell_thresholds)
    count = 0
    
    for buy_t, sell_t in product(buy_thresholds, sell_thresholds):
        count += 1
        print(f"\rTest {count}/{total}: Buy={buy_t:+.1f}%, Sell={sell_t:+.1f}%", end='')
        
        result = simulate_strategy(df, model, scaler, buy_t, sell_t)
        results.append(result)
    
    print("\n")
    
    # Tri par ROI
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print("\n" + "=" * 70)
    print("TOP 10 COMBINAISONS")
    print("=" * 70)
    print(f"{'Buy':>6} | {'Sell':>6} | {'ROI':>8} | {'Trades':>7} | {'Capital':>12}")
    print("-" * 70)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{r['buy_threshold']:>5.1f}% | {r['sell_threshold']:>5.1f}% | "
              f"{r['roi']:>7.2f}% | {r['trades']:>7} | ${r['final_capital']:>11,.0f}")
    
    # Meilleur compromis (ROI > 0 et trades raisonnables)
    print("\n" + "=" * 70)
    print("RECOMMANDATION")
    print("=" * 70)
    
    # Filtre : trades entre 10 et 100
    good_results = [r for r in results if 10 <= r['trades'] <= 100 and r['roi'] > 0]
    
    if good_results:
        best = good_results[0]
        print(f"\nMeilleurs paramètres (ROI max avec trades raisonnables) :")
        print(f"  Buy threshold  : {best['buy_threshold']:+.1f}%")
        print(f"  Sell threshold : {best['sell_threshold']:+.1f}%")
        print(f"  ROI attendu    : {best['roi']:+.2f}%")
        print(f"  Nombre trades  : {best['trades']}")
        print(f"  Capital final  : ${best['final_capital']:,.0f}")
    else:
        print("\nAucune combinaison rentable trouvée avec un nombre raisonnable de trades")
        print("\nMeilleur ROI absolu :")
        best = results[0]
        print(f"  Buy threshold  : {best['buy_threshold']:+.1f}%")
        print(f"  Sell threshold : {best['sell_threshold']:+.1f}%")
        print(f"  ROI            : {best['roi']:+.2f}%")
        print(f"  Trades         : {best['trades']}")

if __name__ == "__main__":
    main()