import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import math

# Configuration
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
    """Charge le modèle et les métadonnées"""
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
    """Prépare les features pour la prédiction"""
    feature_cols = [
        'price', 'volume', 'open', 'high', 'low',
        'ma_7', 'ma_25', 'ma_99', 'rsi', 'macd', 'macd_signal',
        'volatility', 'price_change', 'price_change_1h', 'price_change_24h',
        'volume_change', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower'
    ]
    return df[feature_cols].values

def make_prediction(model, scaler, sequence):
    """Fait une prédiction"""
    sequence_scaled = scaler.transform(sequence)
    sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(sequence_tensor)
    
    return prediction.cpu().numpy()[0]

class TradingBot:
    """Bot de trading avec stratégie basée sur les prédictions"""
    
    def __init__(self, initial_capital=100000, fee_rate=0.001, 
                 buy_threshold=0.5, sell_threshold=-0.3):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # Nombre de SOL possédés
        self.fee_rate = fee_rate  # Frais de transaction (0.1%)
        self.buy_threshold = buy_threshold  # Acheter si prédiction > +0.5%
        self.sell_threshold = sell_threshold  # Vendre si prédiction < -0.3%
        
        self.trades = []
        self.portfolio_value = []
        self.timestamps = []
        
    def execute_trade(self, timestamp, current_price, predicted_change, action=None):
        """Exécute un trade"""
        portfolio_before = self.capital + (self.position * current_price)
        
        if action == 'buy' and self.capital > 0:
            # Acheter avec 50% du capital disponible
            amount_to_invest = self.capital * 0.5
            fee = amount_to_invest * self.fee_rate
            sol_bought = (amount_to_invest - fee) / current_price
            
            self.position += sol_bought
            self.capital -= amount_to_invest
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': current_price,
                'amount': sol_bought,
                'capital': self.capital,
                'position': self.position,
                'predicted_change': predicted_change,
                'fee': fee
            })
            
        elif action == 'sell' and self.position > 0:
            # Vendre 50% de la position
            sol_to_sell = self.position * 0.5
            revenue = sol_to_sell * current_price
            fee = revenue * self.fee_rate
            
            self.position -= sol_to_sell
            self.capital += (revenue - fee)
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': current_price,
                'amount': sol_to_sell,
                'capital': self.capital,
                'position': self.position,
                'predicted_change': predicted_change,
                'fee': fee
            })
        
        portfolio_after = self.capital + (self.position * current_price)
        self.portfolio_value.append(portfolio_after)
        self.timestamps.append(timestamp)
        
        return portfolio_after
    
    def get_signal(self, predicted_change):
        """Détermine l'action à prendre basée sur la prédiction"""
        if predicted_change > self.buy_threshold:
            return 'buy'
        elif predicted_change < self.sell_threshold:
            return 'sell'
        return 'hold'

def run_backtest(df, model, scaler, sequence_length=72, horizon='1h'):
    """Lance le backtesting"""
    print("\n" + "=" * 70)
    print(f"🎯 BACKTESTING - Horizon: {horizon}")
    print("=" * 70)
    
    # Initialisation
    bot = TradingBot(initial_capital=100000)
    
    # Index de la colonne cible selon l'horizon
    horizon_idx = {'1h': 0, '6h': 1, '24h': 2}[horizon]
    
    # Prépare les features
    features = prepare_features(df)
    
    # Nettoyage
    features = np.where(np.isinf(features), np.nan, features)
    mask = ~np.any(np.isnan(features), axis=1)
    features = features[mask]
    df_clean = df[mask].copy()
    
    print(f"📊 Données nettoyées : {len(df_clean)} points")
    print(f"📅 Période : {df_clean.index[0]} → {df_clean.index[-1]}")
    
    # Backtesting
    total_trades = 0
    
    for i in range(sequence_length, len(features) - 1):
        sequence = features[i-sequence_length:i]
        current_price = df_clean.iloc[i]['price']
        timestamp = df_clean.index[i]
        
        # Prédiction
        predictions = make_prediction(model, scaler, sequence)
        predicted_change = predictions[horizon_idx]
        
        # Signal de trading
        signal = bot.get_signal(predicted_change)
        
        # Exécution
        if signal in ['buy', 'sell']:
            bot.execute_trade(timestamp, current_price, predicted_change, signal)
            total_trades += 1
        else:
            # Mise à jour du portfolio même sans trade
            portfolio_value = bot.capital + (bot.position * current_price)
            bot.portfolio_value.append(portfolio_value)
            bot.timestamps.append(timestamp)
    
    # Liquidation finale
    final_price = df_clean.iloc[-1]['price']
    final_timestamp = df_clean.index[-1]
    if bot.position > 0:
        revenue = bot.position * final_price
        fee = revenue * bot.fee_rate
        bot.capital += (revenue - fee)
        bot.position = 0
    
    final_value = bot.capital
    
    print(f"\n💼 RÉSULTATS FINAUX")
    print(f"Capital initial : ${bot.initial_capital:,.2f}")
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Profit/Perte    : ${final_value - bot.initial_capital:,.2f}")
    print(f"ROI             : {((final_value / bot.initial_capital) - 1) * 100:.2f}%")
    print(f"Nombre de trades: {total_trades}")
    
    return bot

def calculate_metrics(bot, df):
    """Calcule les métriques de performance"""
    portfolio_values = np.array(bot.portfolio_value)
    
    # Buy & Hold pour comparaison
    initial_price = df.iloc[0]['price']
    final_price = df.iloc[-1]['price']
    buy_hold_return = ((final_price / initial_price) - 1) * 100
    
    # Returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Sharpe Ratio (annualisé, supposant données horaires)
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24)
    else:
        sharpe_ratio = 0
    
    # Maximum Drawdown
    cumulative = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative) / cumulative
    max_drawdown = np.min(drawdown) * 100
    
    # Win rate
    trades_df = pd.DataFrame(bot.trades)
    if len(trades_df) > 0:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        winning_trades = 0
        total_pairs = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_pairs):
            if i < len(sell_trades) and sell_trades.iloc[i]['price'] > buy_trades.iloc[i]['price']:
                winning_trades += 1
        
        win_rate = (winning_trades / total_pairs * 100) if total_pairs > 0 else 0
    else:
        win_rate = 0
    
    print(f"\n📊 MÉTRIQUES DE PERFORMANCE")
    print(f"Buy & Hold ROI    : {buy_hold_return:+.2f}%")
    print(f"Stratégie ROI     : {((bot.capital / bot.initial_capital) - 1) * 100:+.2f}%")
    print(f"Sharpe Ratio      : {sharpe_ratio:.2f}")
    print(f"Max Drawdown      : {max_drawdown:.2f}%")
    print(f"Win Rate          : {win_rate:.2f}%")
    
    return {
        'buy_hold_roi': buy_hold_return,
        'strategy_roi': ((bot.capital / bot.initial_capital) - 1) * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

def plot_results(bot, df, horizon):
    """Visualise les résultats du backtesting"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Portfolio value
    ax1 = axes[0]
    ax1.plot(bot.timestamps, bot.portfolio_value, label='Portfolio Value', linewidth=2, color='blue')
    ax1.axhline(y=bot.initial_capital, color='gray', linestyle='--', label='Initial Capital')
    ax1.set_title(f'Portfolio Value Over Time - {horizon}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Prix SOL avec trades
    ax2 = axes[1]
    ax2.plot(df.index, df['price'], label='SOL Price', linewidth=1, alpha=0.7, color='orange')
    
    trades_df = pd.DataFrame(bot.trades)
    if len(trades_df) > 0:
        buys = trades_df[trades_df['action'] == 'BUY']
        sells = trades_df[trades_df['action'] == 'SELL']
        
        if len(buys) > 0:
            ax2.scatter(buys['timestamp'], buys['price'], color='green', marker='^', 
                       s=100, label='Buy', zorder=5)
        if len(sells) > 0:
            ax2.scatter(sells['timestamp'], sells['price'], color='red', marker='v', 
                       s=100, label='Sell', zorder=5)
    
    ax2.set_title('SOL Price with Trading Signals', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[2]
    portfolio_values = np.array(bot.portfolio_value)
    cumulative = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative) / cumulative * 100
    ax3.fill_between(bot.timestamps, drawdown, 0, alpha=0.3, color='red')
    ax3.plot(bot.timestamps, drawdown, linewidth=1, color='darkred')
    ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'data/backtest_{horizon}.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé : data/backtest_{horizon}.png")
    plt.close()

def main():
    print("=" * 70)
    print("🤖 BACKTESTING DU BOT DE TRADING")
    print("=" * 70)
    
    # Chargement
    print("\n📂 Chargement du modèle...")
    model, scaler, metadata = load_model()
    
    print("📂 Chargement des données...")
    df = pd.read_csv('data/solana_prepared.csv', index_col='timestamp', parse_dates=True)
    
    # Test sur les 3 horizons
    results = {}
    for horizon in ['1h', '6h', '24h']:
        bot = run_backtest(df, model, scaler, sequence_length=72, horizon=horizon)
        metrics = calculate_metrics(bot, df)
        plot_results(bot, df, horizon)
        results[horizon] = metrics
    
    # Comparaison finale
    print("\n" + "=" * 70)
    print("📊 COMPARAISON DES STRATÉGIES")
    print("=" * 70)
    for horizon, metrics in results.items():
        print(f"\n{horizon}:")
        print(f"  ROI Stratégie : {metrics['strategy_roi']:+.2f}%")
        print(f"  ROI Buy&Hold  : {metrics['buy_hold_roi']:+.2f}%")
        print(f"  Différence    : {metrics['strategy_roi'] - metrics['buy_hold_roi']:+.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ BACKTESTING TERMINÉ")
    print("=" * 70)

if __name__ == "__main__":
    main()