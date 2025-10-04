import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import math

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

class TradingBot:
    def __init__(self, initial_capital=100000, fee_rate=0.003, 
                 buy_threshold=1.5, sell_threshold=-1.0, max_trades_per_day=3):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.fee_rate = fee_rate  # 0.3% (plus réaliste)
        self.buy_threshold = buy_threshold  # Plus conservateur
        self.sell_threshold = sell_threshold
        self.max_trades_per_day = max_trades_per_day
        
        self.trades = []
        self.portfolio_value = []
        self.timestamps = []
        self.daily_trades = {}
        
    def can_trade_today(self, timestamp):
        """Vérifie si on peut encore trader aujourd'hui"""
        date = timestamp.date()
        trades_today = self.daily_trades.get(date, 0)
        return trades_today < self.max_trades_per_day
    
    def execute_trade(self, timestamp, current_price, predicted_change, action=None):
        portfolio_before = self.capital + (self.position * current_price)
        
        date = timestamp.date()
        
        if action == 'buy' and self.capital > 0 and self.can_trade_today(timestamp):
            # Investir 30% du capital (plus prudent)
            amount_to_invest = self.capital * 0.3
            fee = amount_to_invest * self.fee_rate
            sol_bought = (amount_to_invest - fee) / current_price
            
            self.position += sol_bought
            self.capital -= amount_to_invest
            
            self.daily_trades[date] = self.daily_trades.get(date, 0) + 1
            
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
            
        elif action == 'sell' and self.position > 0 and self.can_trade_today(timestamp):
            # Vendre 40% de la position
            sol_to_sell = self.position * 0.4
            revenue = sol_to_sell * current_price
            fee = revenue * self.fee_rate
            
            self.position -= sol_to_sell
            self.capital += (revenue - fee)
            
            self.daily_trades[date] = self.daily_trades.get(date, 0) + 1
            
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
        if predicted_change > self.buy_threshold:
            return 'buy'
        elif predicted_change < self.sell_threshold:
            return 'sell'
        return 'hold'

def run_backtest(df, model, scaler, test_start_date='2024-09-01', 
                 sequence_length=72, horizon='24h'):
    print("\n" + "=" * 70)
    print(f"BACKTESTING RÉALISTE - Horizon: {horizon}")
    print("=" * 70)
    
    # Séparation train/test
    df_test = df[df.index >= test_start_date].copy()
    
    if len(df_test) < sequence_length:
        print(f"Pas assez de données de test après {test_start_date}")
        return None
    
    print(f"Période de test : {df_test.index[0]} → {df_test.index[-1]}")
    print(f"Nombre de points de test : {len(df_test)}")
    
    # Initialisation
    bot = TradingBot(
        initial_capital=100000,
        fee_rate=0.003,  # 0.3%
        buy_threshold=1.5,  # Plus conservateur
        sell_threshold=-1.0,
        max_trades_per_day=3
    )
    
    horizon_idx = {'1h': 0, '6h': 1, '24h': 2}[horizon]
    
    # Préparation des features sur TOUTES les données (pour avoir l'historique)
    features = prepare_features(df)
    features = np.where(np.isinf(features), np.nan, features)
    mask = ~np.any(np.isnan(features), axis=1)
    features = features[mask]
    df_clean = df[mask].copy()
    
    # Trouve l'index le plus proche de la date de début du test
    test_start_idx = df_clean.index.searchsorted(df_test.index[0])
    
    total_trades = 0
    
    # Backtesting uniquement sur la période de test
    for i in range(test_start_idx, len(features) - 1):
        if i < sequence_length:
            continue
            
        sequence = features[i-sequence_length:i]
        current_price = df_clean.iloc[i]['price']
        timestamp = df_clean.index[i]
        
        # Prédiction
        predictions = make_prediction(model, scaler, sequence)
        predicted_change = predictions[horizon_idx]
        
        # Signal
        signal = bot.get_signal(predicted_change)
        
        # Exécution
        if signal in ['buy', 'sell']:
            bot.execute_trade(timestamp, current_price, predicted_change, signal)
            total_trades += 1
        else:
            portfolio_value = bot.capital + (bot.position * current_price)
            bot.portfolio_value.append(portfolio_value)
            bot.timestamps.append(timestamp)
    
    # Liquidation finale
    final_price = df_clean.iloc[-1]['price']
    if bot.position > 0:
        revenue = bot.position * final_price
        fee = revenue * bot.fee_rate
        bot.capital += (revenue - fee)
        bot.position = 0
    
    final_value = bot.capital
    
    print(f"\nRÉSULTATS FINAUX")
    print(f"Capital initial : ${bot.initial_capital:,.2f}")
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Profit/Perte    : ${final_value - bot.initial_capital:,.2f}")
    print(f"ROI             : {((final_value / bot.initial_capital) - 1) * 100:.2f}%")
    print(f"Nombre de trades: {total_trades}")
    print(f"Trades par jour : {total_trades / max(1, (df_test.index[-1] - df_test.index[0]).days):.2f}")
    
    return bot, df_test

def calculate_metrics(bot, df_test):
    portfolio_values = np.array(bot.portfolio_value)
    
    # Buy & Hold
    initial_price = df_test.iloc[0]['price']
    final_price = df_test.iloc[-1]['price']
    buy_hold_return = ((final_price / initial_price) - 1) * 100
    
    # Returns
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe Ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        cumulative = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative) / cumulative
        max_drawdown = np.min(drawdown) * 100
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    # Total des frais
    trades_df = pd.DataFrame(bot.trades)
    total_fees = trades_df['fee'].sum() if len(trades_df) > 0 else 0
    
    print(f"\nMÉTRIQUES")
    print(f"Buy & Hold ROI    : {buy_hold_return:+.2f}%")
    print(f"Stratégie ROI     : {((bot.capital / bot.initial_capital) - 1) * 100:+.2f}%")
    print(f"Alpha             : {((bot.capital / bot.initial_capital) - 1) * 100 - buy_hold_return:+.2f}%")
    print(f"Sharpe Ratio      : {sharpe_ratio:.2f}")
    print(f"Max Drawdown      : {max_drawdown:.2f}%")
    print(f"Total des frais   : ${total_fees:,.2f}")
    
    return {
        'buy_hold_roi': buy_hold_return,
        'strategy_roi': ((bot.capital / bot.initial_capital) - 1) * 100,
        'alpha': ((bot.capital / bot.initial_capital) - 1) * 100 - buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_fees': total_fees
    }

def plot_results(bot, df_test, horizon):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Portfolio value
    ax1 = axes[0]
    ax1.plot(bot.timestamps, bot.portfolio_value, label='Portfolio Value', linewidth=2, color='blue')
    ax1.axhline(y=bot.initial_capital, color='gray', linestyle='--', label='Initial Capital')
    
    # Buy & Hold pour comparaison
    buy_hold_values = []
    initial_price = df_test.iloc[0]['price']
    shares = bot.initial_capital / initial_price
    for ts in bot.timestamps:
        if ts in df_test.index:
            current_price = df_test.loc[ts]['price']
            buy_hold_values.append(shares * current_price)
    
    if len(buy_hold_values) == len(bot.timestamps):
        ax1.plot(bot.timestamps, buy_hold_values, label='Buy & Hold', 
                linewidth=2, color='orange', linestyle='--', alpha=0.7)
    
    ax1.set_title(f'Portfolio Value - {horizon} (Période de test uniquement)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prix avec trades
    ax2 = axes[1]
    ax2.plot(df_test.index, df_test['price'], label='SOL Price', linewidth=1, alpha=0.7, color='orange')
    
    trades_df = pd.DataFrame(bot.trades)
    if len(trades_df) > 0:
        buys = trades_df[trades_df['action'] == 'BUY']
        sells = trades_df[trades_df['action'] == 'SELL']
        
        if len(buys) > 0:
            ax2.scatter(buys['timestamp'], buys['price'], color='green', marker='^', 
                       s=100, label=f'Buy ({len(buys)})', zorder=5)
        if len(sells) > 0:
            ax2.scatter(sells['timestamp'], sells['price'], color='red', marker='v', 
                       s=100, label=f'Sell ({len(sells)})', zorder=5)
    
    ax2.set_title('SOL Price with Trading Signals', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Drawdown
    ax3 = axes[2]
    if len(bot.portfolio_value) > 1:
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
    plt.savefig(f'data/backtest_realistic_{horizon}.png', dpi=150, bbox_inches='tight')
    print(f"\nGraphique sauvegardé : data/backtest_realistic_{horizon}.png")
    plt.close()

def main():
    print("=" * 70)
    print("BACKTESTING RÉALISTE (Sans Data Leakage)")
    print("=" * 70)
    print("\nAVERTISSEMENT : Ce backtesting utilise :")
    print("- Frais réalistes (0.3%)")
    print("- Limite de 3 trades/jour")
    print("- Test uniquement sur données non vues (sept 2024+)")
    print("- Positions partielles (30% achat, 40% vente)")
    
    # Chargement
    print("\nChargement du modèle...")
    model, scaler, metadata = load_model()
    
    print("Chargement des données...")
    df = pd.read_csv('data/solana_prepared.csv', index_col='timestamp', parse_dates=True)
    
    # Test sur les 3 horizons
    results = {}
    for horizon in ['1h', '6h', '24h']:
        result = run_backtest(df, model, scaler, 
                             test_start_date='2024-09-01',
                             sequence_length=72, 
                             horizon=horizon)
        
        if result:
            bot, df_test = result
            metrics = calculate_metrics(bot, df_test)
            plot_results(bot, df_test, horizon)
            results[horizon] = metrics
    
    # Comparaison
    print("\n" + "=" * 70)
    print("COMPARAISON FINALE")
    print("=" * 70)
    for horizon, metrics in results.items():
        print(f"\n{horizon}:")
        print(f"  Stratégie     : {metrics['strategy_roi']:+.2f}%")
        print(f"  Buy & Hold    : {metrics['buy_hold_roi']:+.2f}%")
        print(f"  Alpha         : {metrics['alpha']:+.2f}%")
        print(f"  Sharpe        : {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown  : {metrics['max_drawdown']:.2f}%")
    
    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)

if __name__ == "__main__":
    main()