import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import time
from datetime import datetime, timedelta
import math
import os
from pathlib import Path

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

class PaperTradingBot:
    """Bot de paper trading en temps réel"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.entry_price = None
        
        # Paramètres optimisés
        self.buy_threshold = 1.5
        self.sell_threshold = -1.5
        self.stop_loss_pct = -0.12
        self.take_profit_pct = 0.25
        self.trade_size_pct = 0.30
        self.fee_rate = 0.003
        
        # Logs
        self.trades = []
        self.portfolio_history = []
        
        # Chargement du modèle
        self.model, self.scaler = self.load_model()
        
        print("Paper Trading Bot initialisé")
        print(f"Capital initial : ${self.capital:,.2f}")
        print(f"Seuils : Buy={self.buy_threshold}%, Sell={self.sell_threshold}%")

    def save_state(self):
        """Sauvegarde l'état du bot"""
        logs_dir = Path('logs')
        logs_dir.mkdir(parents=True, exist_ok=True)

        state = {
            'capital': self.capital,
            'position': self.position,
            'entry_price': self.entry_price,
            'trades': self.trades,
            'timestamp': datetime.now().isoformat()
        }

        state_file = logs_dir / 'bot_state.json'
        with state_file.open('w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, default=str)
    
    def load_model(self):
        """Charge le modèle"""
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
        
        return model, scaler
    
    def fetch_live_data(self):
        """Récupère les données en temps réel"""
        sol = yf.Ticker("SOL-USD")
        df = sol.history(period='30d', interval='1h')
        
        if df.empty:
            return None
        
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'price',
            'Volume': 'volume'
        })
        
        df = df[['timestamp', 'open', 'high', 'low', 'price', 'volume']]
        df.set_index('timestamp', inplace=True)
        
        return self.add_indicators(df)
    
    def add_indicators(self, df):
        df['ma_7'] = df['price'].rolling(window=7).mean()
        df['ma_25'] = df['price'].rolling(window=25).mean()
        df['ma_99'] = df['price'].rolling(window=99).mean()
    
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
        df['volatility'] = df['price'].rolling(window=24).std()
        df['price_change'] = df['price'].pct_change()
        df['price_change_1h'] = df['price'].pct_change(periods=1)
        df['price_change_24h'] = df['price'].pct_change(periods=24)
        df['volume_change'] = df['volume'].pct_change()
    
        df['bb_middle'] = df['price'].rolling(window=20).mean()
        df['bb_std'] = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

        # Nettoyage final
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    
    def make_prediction(self, df):
        """Fait une prédiction"""
        feature_cols = [
            'price', 'volume', 'open', 'high', 'low',
            'ma_7', 'ma_25', 'ma_99', 'rsi', 'macd', 'macd_signal',
            'volatility', 'price_change', 'price_change_1h', 'price_change_24h',
            'volume_change', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower'
        ]
        
        sequence = df[feature_cols].tail(72).values
        sequence_scaled = self.scaler.transform(sequence)
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
        
        return prediction.cpu().numpy()[0]
    
    def check_stop_loss_take_profit(self, current_price):
        """Vérifie SL/TP"""
        if self.position > 0 and self.entry_price:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            if pnl_pct <= self.stop_loss_pct:
                return 'stop_loss'
            elif pnl_pct >= self.take_profit_pct:
                return 'take_profit'
        
        return None
    
    def execute_trade(self, action, current_price, predicted_change, current_rsi):
        """Exécute un trade"""
        timestamp = datetime.now()
        
        if action == 'buy' and self.capital > 0 and self.position == 0:
            amount = self.capital * self.trade_size_pct
            fee = amount * self.fee_rate
            self.position = (amount - fee) / current_price
            self.capital -= amount
            self.entry_price = current_price
            
            trade = {
                'timestamp': timestamp,
                'action': 'BUY',
                'price': current_price,
                'amount': self.position,
                'capital': self.capital,
                'predicted_change': predicted_change,
                'rsi': current_rsi,
                'fee': fee
            }
            
            self.trades.append(trade)
            self.log_trade(trade)
            
        elif action in ['sell', 'stop_loss', 'take_profit'] and self.position > 0:
            revenue = self.position * current_price
            fee = revenue * self.fee_rate
            self.capital += (revenue - fee)
            
            pnl = revenue - (self.entry_price * self.position) - fee
            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            
            trade = {
                'timestamp': timestamp,
                'action': action.upper(),
                'price': current_price,
                'amount': self.position,
                'capital': self.capital,
                'predicted_change': predicted_change,
                'rsi': current_rsi,
                'fee': fee,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            
            self.position = 0
            self.entry_price = None
            
            self.trades.append(trade)
            self.log_trade(trade)
    
    def log_trade(self, trade):
        """Enregistre un trade dans le fichier log"""
        os.makedirs('logs', exist_ok=True)
        
        with open('logs/paper_trading.log', 'a') as f:
            f.write(f"\n{'-'*60}\n")
            f.write(f"[{trade['timestamp']}]\n")
            f.write(f"Action: {trade['action']}\n")
            f.write(f"Price: ${trade['price']:.2f}\n")
            f.write(f"Amount: {trade['amount']:.4f} SOL\n")
            f.write(f"Capital: ${trade['capital']:,.2f}\n")
            f.write(f"Prediction: {trade['predicted_change']:+.2f}%\n")
            f.write(f"RSI: {trade['rsi']:.1f}\n")
            
            if 'pnl' in trade:
                f.write(f"P/L: ${trade['pnl']:+,.2f} ({trade['pnl_pct']:+.2f}%)\n")
    
    
    def get_status(self, current_price):
        """Affiche le statut actuel"""
        portfolio_value = self.capital + (self.position * current_price)
        roi = ((portfolio_value / self.initial_capital) - 1) * 100
        
        print(f"\n{'='*60}")
        print(f"STATUT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Prix SOL        : ${current_price:.2f}")
        print(f"Capital cash    : ${self.capital:,.2f}")
        print(f"Position        : {self.position:.4f} SOL (${self.position * current_price:,.2f})")
        print(f"Portfolio total : ${portfolio_value:,.2f}")
        print(f"ROI             : {roi:+.2f}%")
        print(f"Trades exécutés : {len(self.trades)}")
        
        if self.entry_price and self.position > 0:
            unrealized_pnl = ((current_price - self.entry_price) / self.entry_price) * 100
            print(f"P/L non réalisé : {unrealized_pnl:+.2f}%")
    
    def run(self, check_interval=3600):
        """Lance le bot en continu"""
        print(f"\n{'='*60}")
        print("PAPER TRADING BOT DÉMARRÉ")
        print(f"{'='*60}")
        print(f"Intervalle de vérification : {check_interval}s ({check_interval/3600:.1f}h)")
        print("Appuyez sur Ctrl+C pour arrêter\n")
        
        try:
            while True:
                # Récupération des données
                df = self.fetch_live_data()
                
                if df is None or len(df) < 72:
                    print("Pas assez de données, attente...")
                    time.sleep(check_interval)
                    continue
                
                current_price = df['price'].iloc[-1]
                current_rsi = df['rsi'].iloc[-1]
                
                # Prédiction
                predictions = self.make_prediction(df)
                pred_24h = predictions[2]
                
                # Check SL/TP
                auto_action = self.check_stop_loss_take_profit(current_price)
                
                if auto_action:
                    print(f"\n⚠️  {auto_action.upper()} déclenché !")
                    self.execute_trade(auto_action, current_price, pred_24h, current_rsi)
                
                # Signal de trading
                elif pred_24h > self.buy_threshold and self.position == 0:
                    print(f"\n📈 SIGNAL D'ACHAT (prédiction: {pred_24h:+.2f}%)")
                    self.execute_trade('buy', current_price, pred_24h, current_rsi)
                
                elif pred_24h < self.sell_threshold and self.position > 0:
                    print(f"\n📉 SIGNAL DE VENTE (prédiction: {pred_24h:+.2f}%)")
                    self.execute_trade('sell', current_price, pred_24h, current_rsi)
                
                else:
                    print(f"\n⏸️  HOLD (prédiction: {pred_24h:+.2f}%)")
                
                # Statut
                self.get_status(current_price)
                
                # Sauvegarde
                self.save_state()
                
                # Attente
                print(f"\nProchaine vérification dans {check_interval/3600:.1f}h...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n{'='*60}")
            print("BOT ARRÊTÉ PAR L'UTILISATEUR")
            print(f"{'='*60}")
            self.save_state()
            self.get_status(current_price)
            print("\nÉtat sauvegardé dans logs/")

def main():
    bot = PaperTradingBot(initial_capital=100000)
    bot.run(check_interval=3600)  # Vérifie toutes les heures

if __name__ == "__main__":
    main()