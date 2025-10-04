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

class FinalTradingBot:
    """Bot avec tous les bugs corrigés"""
    
    def __init__(self, initial_capital=100000, fee_rate=0.003):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.fee_rate = fee_rate
        
        # Seuils optimisés
        self.buy_threshold = 1.5
        self.sell_threshold = -1.5
        
        # Risk management
        self.min_position_for_tp = 0.1  # Minimum 10% de position pour take profit
        self.trade_size_pct = 0.30
        self.stop_loss_pct = -0.12
        self.take_profit_pct = 0.25
        
        # Limites strictes
        self.max_trades_per_day = 2
        self.min_hours_between_trades = 6
        
        # Tracking
        self.trades = []
        self.portfolio_value = []
        self.timestamps = []
        self.entry_price = None
        self.last_trade_time = None
        self.daily_trade_count = {}
        
    def can_trade_now(self, timestamp):
        """Vérifie si on peut trader (limite journalière ET temporelle)"""
        date = timestamp.date()
        trades_today = self.daily_trade_count.get(date, 0)
        
        # Limite journalière
        if trades_today >= self.max_trades_per_day:
            return False
        
        # Limite temporelle (6h minimum entre trades)
        if self.last_trade_time:
            hours_since_last = (timestamp - self.last_trade_time).total_seconds() / 3600
            if hours_since_last < self.min_hours_between_trades:
                return False
        
        return True
    
    def check_stop_loss_take_profit(self, current_price):
        """Vérifie SL/TP uniquement si position significative"""
        if self.position > 0 and self.entry_price:
            # Calcul de la valeur de position
            position_value = self.position * current_price
            total_value = self.capital + position_value
            position_pct = position_value / total_value if total_value > 0 else 0
            
            # Ignore les micro-positions
            if position_pct < self.min_position_for_tp:
                return None
            
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            if pnl_pct <= self.stop_loss_pct:
                return 'stop_loss'
            elif pnl_pct >= self.take_profit_pct:
                return 'take_profit'
        
        return None
    
    def get_signal(self, predicted_change, current_rsi, current_price, timestamp):
        """Signal de trading simplifié"""
        
        # Check SL/TP d'abord (sans limite de trades)
        auto_signal = self.check_stop_loss_take_profit(current_price)
        if auto_signal:
            return auto_signal
        
        # Vérifie les limites de trading
        if not self.can_trade_now(timestamp):
            return 'hold'
        
        # Position actuelle
        position_value = self.position * current_price
        total_value = self.capital + position_value
        position_pct = position_value / total_value if total_value > 0 else 0
        
        # Signaux d'achat
        if predicted_change > self.buy_threshold:
            if current_rsi < 70 and position_pct < 0.7 and self.capital > 1000:
                return 'buy'
        
        # Signaux de vente
        elif predicted_change < self.sell_threshold:
            if current_rsi > 30 and self.position > 0:
                return 'sell'
        
        return 'hold'
    
    def execute_trade(self, timestamp, current_price, current_rsi, predicted_change, signal):
        """Exécute un trade"""
        
        if signal == 'buy':
            if self.capital <= 0:
                return self._update_portfolio_value(timestamp, current_price)
            
            amount_to_invest = self.capital * self.trade_size_pct
            fee = amount_to_invest * self.fee_rate
            sol_bought = (amount_to_invest - fee) / current_price
            
            self.position += sol_bought
            self.capital -= amount_to_invest
            self.entry_price = current_price
            self.last_trade_time = timestamp
            
            date = timestamp.date()
            self.daily_trade_count[date] = self.daily_trade_count.get(date, 0) + 1
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': current_price,
                'amount': sol_bought,
                'capital': self.capital,
                'position': self.position,
                'predicted_change': predicted_change,
                'rsi': current_rsi,
                'fee': fee,
                'reason': 'signal'
            })
            
        elif signal in ['sell', 'stop_loss', 'take_profit']:
            if self.position <= 0:
                return self._update_portfolio_value(timestamp, current_price)
            
            # Vente complète pour simplifier
            sol_to_sell = self.position
            revenue = sol_to_sell * current_price
            fee = revenue * self.fee_rate
            
            self.position = 0
            self.capital += (revenue - fee)
            self.entry_price = None
            
            # Compte comme trade seulement si signal normal (pas auto SL/TP)
            if signal == 'sell':
                self.last_trade_time = timestamp
                date = timestamp.date()
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
        portfolio_value = self.capital + (self.position * current_price)
        self.portfolio_value.append(portfolio_value)
        self.timestamps.append(timestamp)
        return portfolio_value

def run_final_backtest(df, model, scaler, test_start_date='2024-09-01', sequence_length=72):
    print("\n" + "=" * 70)
    print("BACKTESTING FINAL - Tous bugs corrigés")
    print("=" * 70)
    
    df_test = df[df.index >= test_start_date].copy()
    
    if len(df_test) < sequence_length:
        print(f"Pas assez de données")
        return None
    
    print(f"Période : {df_test.index[0]} → {df_test.index[-1]}")
    print(f"Points : {len(df_test)}")
    
    bot = FinalTradingBot(initial_capital=100000, fee_rate=0.003)
    
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
        
        predictions = make_prediction(model, scaler, sequence)
        predicted_change = predictions[2]
        
        signal = bot.get_signal(predicted_change, current_rsi, current_price, timestamp)
        
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
    
    print(f"\nRÉSULTATS")
    print(f"Capital initial : ${bot.initial_capital:,.2f}")
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Profit/Perte    : ${final_value - bot.initial_capital:,.2f}")
    print(f"ROI             : {((final_value / bot.initial_capital) - 1) * 100:.2f}%")
    print(f"Trades          : {len(bot.trades)}")
    
    return bot, df_test

def analyze_final_trades(bot):
    """Analyse détaillée"""
    trades_df = pd.DataFrame(bot.trades)
    
    if len(trades_df) == 0:
        print("\nAucun trade")
        return
    
    print(f"\n" + "=" * 70)
    print("ANALYSE DÉTAILLÉE")
    print("=" * 70)
    
    print(f"\nActions :")
    print(trades_df['action'].value_counts())
    
    print(f"\nRaisons :")
    print(trades_df['reason'].value_counts())
    
    total_fees = trades_df['fee'].sum()
    print(f"\nFrais totaux : ${total_fees:,.2f}")
    
    # Trades par jour
    trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
    trades_per_day = trades_df.groupby('date').size()
    print(f"\nTrades par jour :")
    print(f"  Moyen : {trades_per_day.mean():.2f}")
    print(f"  Max : {trades_per_day.max()}")
    print(f"  Jours avec trades : {len(trades_per_day)}")
    
    # Win rate
    buys = trades_df[trades_df['action'] == 'BUY']
    sells = trades_df[trades_df['action'].isin(['SELL', 'STOP_LOSS', 'TAKE_PROFIT'])]
    
    if len(buys) > 0 and len(sells) > 0:
        wins = 0
        losses = 0
        
        for i in range(min(len(buys), len(sells))):
            if sells.iloc[i]['price'] > buys.iloc[i]['price']:
                wins += 1
            else:
                losses += 1
        
        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0
        print(f"\nWin Rate : {win_rate:.1f}%")
        print(f"  Gagnants : {wins}")
        print(f"  Perdants : {losses}")

def calculate_final_metrics(bot, df_test):
    portfolio_values = np.array(bot.portfolio_value)
    
    initial_price = df_test.iloc[0]['price']
    final_price = df_test.iloc[-1]['price']
    buy_hold_return = ((final_price / initial_price) - 1) * 100
    
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
        
        cumulative = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative) / cumulative
        max_drawdown = np.min(drawdown) * 100
        
        annual_return = ((bot.capital / bot.initial_capital) - 1) * 100
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    else:
        sharpe_ratio = 0
        max_drawdown = 0
        calmar_ratio = 0
    
    print(f"\n" + "=" * 70)
    print("MÉTRIQUES")
    print("=" * 70)
    print(f"Buy & Hold      : {buy_hold_return:+.2f}%")
    print(f"Stratégie       : {((bot.capital / bot.initial_capital) - 1) * 100:+.2f}%")
    print(f"Alpha           : {((bot.capital / bot.initial_capital) - 1) * 100 - buy_hold_return:+.2f}%")
    print(f"Sharpe Ratio    : {sharpe_ratio:.2f}")
    print(f"Max Drawdown    : {max_drawdown:.2f}%")
    print(f"Calmar Ratio    : {calmar_ratio:.2f}")

def plot_final_results(bot, df_test):
    """Graphiques finaux"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Portfolio
    ax1 = axes[0]
    ax1.plot(bot.timestamps, bot.portfolio_value, label='Stratégie', 
            linewidth=2.5, color='#2E86AB')
    ax1.axhline(y=bot.initial_capital, color='gray', linestyle='--', label='Initial', linewidth=1.5)
    
    initial_price = df_test.iloc[0]['price']
    shares = bot.initial_capital / initial_price
    buy_hold = [shares * df_test.loc[ts]['price'] for ts in bot.timestamps if ts in df_test.index]
    
    if len(buy_hold) == len(bot.timestamps):
        ax1.plot(bot.timestamps, buy_hold, label='Buy & Hold', 
                linewidth=2, color='orange', linestyle='--', alpha=0.8)
    
    ax1.set_title('Portfolio Value', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value (USD)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Prix + trades
    ax2 = axes[1]
    ax2.plot(df_test.index, df_test['price'], linewidth=1.5, alpha=0.7, color='#FF6B35')
    
    trades_df = pd.DataFrame(bot.trades)
    if len(trades_df) > 0:
        buys = trades_df[trades_df['action'] == 'BUY']
        sells = trades_df[trades_df['reason'] == 'sell']
        stops = trades_df[trades_df['reason'] == 'stop_loss']
        takes = trades_df[trades_df['reason'] == 'take_profit']
        
        if len(buys) > 0:
            ax2.scatter(buys['timestamp'], buys['price'], color='green', 
                       marker='^', s=200, label=f'Buy ({len(buys)})', zorder=5)
        if len(sells) > 0:
            ax2.scatter(sells['timestamp'], sells['price'], color='red', 
                       marker='v', s=200, label=f'Sell ({len(sells)})', zorder=5)
        if len(stops) > 0:
            ax2.scatter(stops['timestamp'], stops['price'], color='darkred', 
                       marker='X', s=250, label=f'Stop Loss ({len(stops)})', zorder=6)
        if len(takes) > 0:
            ax2.scatter(takes['timestamp'], takes['price'], color='darkgreen', 
                       marker='*', s=300, label=f'Take Profit ({len(takes)})', zorder=6)
    
    ax2.set_title('Trading Signals', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USD)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Drawdown
    ax3 = axes[2]
    portfolio_values = np.array(bot.portfolio_value)
    cumulative = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative) / cumulative * 100
    ax3.fill_between(bot.timestamps, drawdown, 0, alpha=0.4, color='#C1121F')
    ax3.plot(bot.timestamps, drawdown, linewidth=2, color='darkred')
    ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/backtest_final.png', dpi=200, bbox_inches='tight')
    print(f"\nGraphique : data/backtest_final.png")
    plt.close()

def main():
    print("=" * 70)
    print("BACKTESTING FINAL - VERSION CORRIGÉE")
    print("=" * 70)
    print("\nCorrections appliquées :")
    print("- Limite stricte : 2 trades/jour MAX")
    print("- Minimum 6h entre chaque trade")
    print("- Take Profit uniquement si position > 10%")
    print("- Seuils plus conservateurs (±3%/±2.5%)")
    print("- Vente complète (pas de micro-positions)")
    
    model, scaler, metadata = load_model()
    df = pd.read_csv('data/solana_prepared.csv', index_col='timestamp', parse_dates=True)
    
    bot, df_test = run_final_backtest(df, model, scaler, test_start_date='2024-09-01')
    
    if bot:
        analyze_final_trades(bot)
        calculate_final_metrics(bot, df_test)
        plot_final_results(bot, df_test)
    
    print("\n" + "=" * 70)
    print("TERMINÉ - Résultats réalistes")
    print("=" * 70)

if __name__ == "__main__":
    main()