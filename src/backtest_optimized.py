import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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

class SmartTradingBot:
    """Bot avec stratégie optimisée et risk management"""
    
    def __init__(self, initial_capital=100000, fee_rate=0.003):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.fee_rate = fee_rate
        
        # Paramètres de stratégie
        self.strong_buy_threshold = 3.0  # Très bullish
        self.buy_threshold = 2.0         # Bullish
        self.sell_threshold = -2.0       # Bearish
        self.strong_sell_threshold = -3.0 # Très bearish
        
        # Risk management
        self.max_position_pct = 0.7      # Max 70% du capital en position
        self.trade_size_pct = 0.25       # Taille de trade : 25% du capital
        self.stop_loss_pct = -0.15       # Stop loss à -15%
        self.take_profit_pct = 0.30      # Take profit à +30%
        
        # Tracking
        self.trades = []
        self.portfolio_value = []
        self.timestamps = []
        self.last_trade_date = None
        self.entry_price = None
        self.max_daily_trades = 2
        self.daily_trade_count = {}
        
    def can_trade_today(self, timestamp):
        """Vérifie si on peut encore trader aujourd'hui"""
        date = timestamp.date()
        return self.daily_trade_count.get(date, 0) < self.max_daily_trades
    
    def check_stop_loss_take_profit(self, current_price, timestamp):
        """Vérifie les conditions de stop loss et take profit"""
        if self.position > 0 and self.entry_price:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            if pnl_pct <= self.stop_loss_pct:
                return 'stop_loss'
            elif pnl_pct >= self.take_profit_pct:
                return 'take_profit'
        
        return None
    
    def get_signal(self, predicted_change, current_rsi, current_price, timestamp):
        """Stratégie de trading améliorée avec multiples filtres"""
        
        # Check stop loss / take profit d'abord
        auto_signal = self.check_stop_loss_take_profit(current_price, timestamp)
        if auto_signal:
            return auto_signal
        
        # Pas de trade si limite journalière atteinte
        if not self.can_trade_today(timestamp):
            return 'hold'
        
        # Calcul de la valeur actuelle de la position
        position_value = self.position * current_price
        total_value = self.capital + position_value
        position_pct = position_value / total_value if total_value > 0 else 0
        
        # Signal d'achat fort
        if predicted_change > self.strong_buy_threshold:
            # Confirmation RSI : pas suracheté
            if current_rsi < 70 and position_pct < self.max_position_pct:
                return 'strong_buy'
        
        # Signal d'achat modéré
        elif predicted_change > self.buy_threshold:
            if current_rsi < 65 and position_pct < self.max_position_pct:
                return 'buy'
        
        # Signal de vente
        elif predicted_change < self.sell_threshold:
            # Confirmation RSI : pas survendu
            if current_rsi > 35 and self.position > 0:
                return 'sell'
        
        # Signal de vente fort
        elif predicted_change < self.strong_sell_threshold:
            if current_rsi > 30 and self.position > 0:
                return 'strong_sell'
        
        return 'hold'
    
    def execute_trade(self, timestamp, current_price, current_rsi, predicted_change, signal):
        """Exécute un trade avec la nouvelle stratégie"""
        
        date = timestamp.date()
        
        if signal in ['strong_buy', 'buy']:
            if self.capital <= 0:
                return self._update_portfolio_value(timestamp, current_price)
            
            # Taille du trade
            size_multiplier = 1.5 if signal == 'strong_buy' else 1.0
            amount_to_invest = min(
                self.capital * self.trade_size_pct * size_multiplier,
                self.capital  # Ne pas dépasser le capital disponible
            )
            
            fee = amount_to_invest * self.fee_rate
            sol_bought = (amount_to_invest - fee) / current_price
            
            self.position += sol_bought
            self.capital -= amount_to_invest
            self.entry_price = current_price
            
            self.daily_trade_count[date] = self.daily_trade_count.get(date, 0) + 1
            
            self.trades.append({
                'timestamp': timestamp,
                'action': signal.upper(),
                'price': current_price,
                'amount': sol_bought,
                'capital': self.capital,
                'position': self.position,
                'predicted_change': predicted_change,
                'rsi': current_rsi,
                'fee': fee,
                'reason': 'signal'
            })
            
        elif signal in ['sell', 'strong_sell', 'stop_loss', 'take_profit']:
            if self.position <= 0:
                return self._update_portfolio_value(timestamp, current_price)
            
            # Taille de la vente
            if signal == 'stop_loss':
                sell_pct = 1.0  # Vendre tout en stop loss
            elif signal == 'take_profit':
                sell_pct = 0.7  # Vendre 70% en take profit
            elif signal == 'strong_sell':
                sell_pct = 0.6
            else:
                sell_pct = 0.4
            
            sol_to_sell = self.position * sell_pct
            revenue = sol_to_sell * current_price
            fee = revenue * self.fee_rate
            
            self.position -= sol_to_sell
            self.capital += (revenue - fee)
            
            if self.position == 0:
                self.entry_price = None
            
            self.daily_trade_count[date] = self.daily_trade_count.get(date, 0) + 1
            
            self.trades.append({
                'timestamp': timestamp,
                'action': signal.upper(),
                'price': current_price,
                'amount': sol_to_sell,
                'capital': self.capital,
                'position': self.position,
                'predicted_change': predicted_change,
                'rsi': current_rsi,
                'fee': fee,
                'reason': signal
            })
        
        return self._update_portfolio_value(timestamp, current_price)
    
    def _update_portfolio_value(self, timestamp, current_price):
        """Met à jour la valeur du portfolio"""
        portfolio_value = self.capital + (self.position * current_price)
        self.portfolio_value.append(portfolio_value)
        self.timestamps.append(timestamp)
        return portfolio_value

def run_optimized_backtest(df, model, scaler, test_start_date='2024-09-01', 
                          sequence_length=72):
    print("\n" + "=" * 70)
    print("BACKTESTING OPTIMISÉ - Horizon 24h")
    print("=" * 70)
    
    df_test = df[df.index >= test_start_date].copy()
    
    if len(df_test) < sequence_length:
        print(f"Pas assez de données de test")
        return None
    
    print(f"Période de test : {df_test.index[0]} → {df_test.index[-1]}")
    print(f"Nombre de points : {len(df_test)}")
    
    bot = SmartTradingBot(initial_capital=100000, fee_rate=0.003)
    
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
        current_rsi = df_clean.iloc[i]['rsi']
        timestamp = df_clean.index[i]
        
        # Prédiction (horizon 24h = index 2)
        predictions = make_prediction(model, scaler, sequence)
        predicted_change = predictions[2]
        
        # Signal
        signal = bot.get_signal(predicted_change, current_rsi, current_price, timestamp)
        
        # Exécution
        if signal != 'hold':
            bot.execute_trade(timestamp, current_price, current_rsi, predicted_change, signal)
        else:
            bot._update_portfolio_value(timestamp, current_price)
    
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
    print(f"Trades exécutés : {len(bot.trades)}")
    
    return bot, df_test

def analyze_trades(bot):
    """Analyse détaillée des trades"""
    trades_df = pd.DataFrame(bot.trades)
    
    if len(trades_df) == 0:
        print("\nAucun trade exécuté")
        return
    
    print(f"\n" + "=" * 70)
    print("ANALYSE DES TRADES")
    print("=" * 70)
    
    # Types de trades
    print(f"\nRépartition des trades :")
    print(trades_df['action'].value_counts())
    
    # Raisons
    print(f"\nRaisons des trades :")
    print(trades_df['reason'].value_counts())
    
    # Frais totaux
    total_fees = trades_df['fee'].sum()
    print(f"\nTotal des frais : ${total_fees:,.2f}")
    
    # Analyse par paires achat/vente
    buys = trades_df[trades_df['action'].str.contains('BUY')]
    sells = trades_df[trades_df['action'].str.contains('SELL|STOP|TAKE')]
    
    print(f"\nAchats : {len(buys)}")
    print(f"Ventes : {len(sells)}")
    
    # Performance moyenne
    if len(buys) > 0 and len(sells) > 0:
        winning_trades = 0
        losing_trades = 0
        
        for i in range(min(len(buys), len(sells))):
            if sells.iloc[i]['price'] > buys.iloc[i]['price']:
                winning_trades += 1
            else:
                losing_trades += 1
        
        win_rate = winning_trades / (winning_trades + losing_trades) * 100
        print(f"\nWin Rate : {win_rate:.1f}%")
        print(f"Trades gagnants : {winning_trades}")
        print(f"Trades perdants : {losing_trades}")

def calculate_advanced_metrics(bot, df_test):
    """Calcule les métriques avancées"""
    portfolio_values = np.array(bot.portfolio_value)
    
    # Buy & Hold
    initial_price = df_test.iloc[0]['price']
    final_price = df_test.iloc[-1]['price']
    buy_hold_return = ((final_price / initial_price) - 1) * 100
    
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
        
        # Calmar Ratio
        annual_return = ((bot.capital / bot.initial_capital) - 1) * 100
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    else:
        sharpe_ratio = 0
        max_drawdown = 0
        calmar_ratio = 0
    
    print(f"\n" + "=" * 70)
    print("MÉTRIQUES DE PERFORMANCE")
    print("=" * 70)
    print(f"Buy & Hold ROI     : {buy_hold_return:+.2f}%")
    print(f"Stratégie ROI      : {((bot.capital / bot.initial_capital) - 1) * 100:+.2f}%")
    print(f"Alpha              : {((bot.capital / bot.initial_capital) - 1) * 100 - buy_hold_return:+.2f}%")
    print(f"Sharpe Ratio       : {sharpe_ratio:.2f}")
    print(f"Max Drawdown       : {max_drawdown:.2f}%")
    print(f"Calmar Ratio       : {calmar_ratio:.2f}")

def plot_optimized_results(bot, df_test):
    """Visualise les résultats"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    # 1. Portfolio value
    ax1 = axes[0]
    ax1.plot(bot.timestamps, bot.portfolio_value, label='Portfolio (Stratégie)', 
            linewidth=2.5, color='#2E86AB')
    ax1.axhline(y=bot.initial_capital, color='gray', linestyle='--', 
               label='Capital initial', linewidth=1.5)
    
    # Buy & Hold
    initial_price = df_test.iloc[0]['price']
    shares = bot.initial_capital / initial_price
    buy_hold_values = [shares * df_test.loc[ts]['price'] 
                      for ts in bot.timestamps if ts in df_test.index]
    
    if len(buy_hold_values) == len(bot.timestamps):
        ax1.plot(bot.timestamps, buy_hold_values, label='Buy & Hold', 
                linewidth=2, color='orange', linestyle='--', alpha=0.8)
    
    ax1.fill_between(bot.timestamps, bot.portfolio_value, bot.initial_capital, 
                     where=np.array(bot.portfolio_value) > bot.initial_capital,
                     alpha=0.2, color='green', label='Profit')
    ax1.fill_between(bot.timestamps, bot.portfolio_value, bot.initial_capital,
                     where=np.array(bot.portfolio_value) < bot.initial_capital,
                     alpha=0.2, color='red', label='Loss')
    
    ax1.set_title('Portfolio Value vs Buy & Hold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value (USD)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Prix avec trades
    ax2 = axes[1]
    ax2.plot(df_test.index, df_test['price'], label='SOL Price', 
            linewidth=1.5, alpha=0.7, color='#FF6B35')
    
    trades_df = pd.DataFrame(bot.trades)
    if len(trades_df) > 0:
        buys = trades_df[trades_df['action'].str.contains('BUY')]
        sells = trades_df[trades_df['action'].str.contains('SELL') & 
                         ~trades_df['reason'].str.contains('stop|take')]
        stops = trades_df[trades_df['reason'] == 'stop_loss']
        takes = trades_df[trades_df['reason'] == 'take_profit']
        
        if len(buys) > 0:
            ax2.scatter(buys['timestamp'], buys['price'], color='#06D6A0', 
                       marker='^', s=150, label=f'Buy ({len(buys)})', 
                       zorder=5, edgecolors='black', linewidths=1)
        if len(sells) > 0:
            ax2.scatter(sells['timestamp'], sells['price'], color='#EF476F', 
                       marker='v', s=150, label=f'Sell ({len(sells)})', 
                       zorder=5, edgecolors='black', linewidths=1)
        if len(stops) > 0:
            ax2.scatter(stops['timestamp'], stops['price'], color='darkred', 
                       marker='X', s=200, label=f'Stop Loss ({len(stops)})', 
                       zorder=6, edgecolors='black', linewidths=1.5)
        if len(takes) > 0:
            ax2.scatter(takes['timestamp'], takes['price'], color='darkgreen', 
                       marker='*', s=250, label=f'Take Profit ({len(takes)})', 
                       zorder=6, edgecolors='black', linewidths=1.5)
    
    ax2.set_title('Trading Signals avec Stop Loss / Take Profit', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USD)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[2]
    portfolio_values = np.array(bot.portfolio_value)
    cumulative = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative) / cumulative * 100
    ax3.fill_between(bot.timestamps, drawdown, 0, alpha=0.4, color='#C1121F')
    ax3.plot(bot.timestamps, drawdown, linewidth=2, color='darkred')
    ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Position exposure
    ax4 = axes[3]
    exposure = []
    for i, ts in enumerate(bot.timestamps):
        position_value = bot.trades[min(i, len(bot.trades)-1)].get('position', 0) * df_test.loc[ts, 'price'] if ts in df_test.index else 0
        total_value = bot.portfolio_value[i]
        exposure.append((position_value / total_value * 100) if total_value > 0 else 0)
    
    ax4.fill_between(bot.timestamps, exposure, 0, alpha=0.3, color='#118AB2')
    ax4.plot(bot.timestamps, exposure, linewidth=2, color='#118AB2')
    ax4.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Max exposure (70%)')
    ax4.set_title('Position Exposure (%)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Exposure (%)', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/backtest_optimized.png', dpi=200, bbox_inches='tight')
    print(f"\nGraphique sauvegardé : data/backtest_optimized.png")
    plt.close()

def main():
    print("=" * 70)
    print("BACKTESTING AVEC STRATÉGIE OPTIMISÉE")
    print("=" * 70)
    print("\nAméliorations :")
    print("- Seuils adaptatifs (±2% et ±3%)")
    print("- Stop Loss automatique (-15%)")
    print("- Take Profit automatique (+30%)")
    print("- Confirmation RSI")
    print("- Max 2 trades/jour")
    print("- Gestion de position (max 70%)")
    
    model, scaler, metadata = load_model()
    df = pd.read_csv('data/solana_prepared.csv', index_col='timestamp', parse_dates=True)
    
    bot, df_test = run_optimized_backtest(df, model, scaler, test_start_date='2024-09-01')
    
    if bot:
        analyze_trades(bot)
        calculate_advanced_metrics(bot, df_test)
        plot_optimized_results(bot, df_test)
    
    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)

if __name__ == "__main__":
    main()