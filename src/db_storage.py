import os
import json
import psycopg2
from datetime import datetime

class DatabaseStorage:
    def __init__(self):
        self.conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        self.init_db()
    
    def init_db(self):
        """Crée les tables si elles n'existent pas"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    capital FLOAT,
                    position FLOAT,
                    entry_price FLOAT,
                    last_update TIMESTAMP,
                    CHECK (id = 1)
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    action TEXT,
                    price FLOAT,
                    amount FLOAT,
                    capital FLOAT,
                    predicted_change FLOAT,
                    rsi FLOAT,
                    fee FLOAT,
                    pnl FLOAT,
                    pnl_pct FLOAT
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    price FLOAT,
                    portfolio_value FLOAT,
                    roi FLOAT
                )
            """)
            
            self.conn.commit()
    
    def save_state(self, capital, position, entry_price):
        """Sauvegarde l'état du bot"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_state (id, capital, position, entry_price, last_update)
                VALUES (1, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    capital = EXCLUDED.capital,
                    position = EXCLUDED.position,
                    entry_price = EXCLUDED.entry_price,
                    last_update = EXCLUDED.last_update
            """, (capital, position, entry_price, datetime.now()))
            self.conn.commit()
    
    def load_state(self):
        """Charge l'état du bot"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT capital, position, entry_price FROM bot_state WHERE id = 1")
            result = cur.fetchone()
            if result:
                return {
                    'capital': result[0],
                    'position': result[1],
                    'entry_price': result[2]
                }
            return None
    
    def add_trade(self, trade):
        """Ajoute un trade"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (timestamp, action, price, amount, capital, 
                                   predicted_change, rsi, fee, pnl, pnl_pct)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                datetime.fromisoformat(trade['timestamp']),
                trade['action'],
                trade['price'],
                trade['amount'],
                trade['capital'],
                trade['predicted_change'],
                trade['rsi'],
                trade['fee'],
                trade.get('pnl', 0),
                trade.get('pnl_pct', 0)
            ))
            self.conn.commit()
    
    def get_trades(self, limit=100):
        """Récupère les derniers trades"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, action, price, amount, capital, 
                       predicted_change, rsi, fee, pnl, pnl_pct
                FROM trades
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            
            trades = []
            for row in cur.fetchall():
                trades.append({
                    'timestamp': row[0].isoformat(),
                    'action': row[1],
                    'price': row[2],
                    'amount': row[3],
                    'capital': row[4],
                    'predicted_change': row[5],
                    'rsi': row[6],
                    'fee': row[7],
                    'pnl': row[8],
                    'pnl_pct': row[9]
                })
            return trades
    
    def add_portfolio_snapshot(self, price, portfolio_value, roi):
        """Ajoute un snapshot du portfolio"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO portfolio_history (timestamp, price, portfolio_value, roi)
                VALUES (%s, %s, %s, %s)
            """, (datetime.now(), price, portfolio_value, roi))
            self.conn.commit()
    
    def get_portfolio_history(self, hours=168):
        """Récupère l'historique du portfolio"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, price, portfolio_value, roi
                FROM portfolio_history
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp ASC
            """, (hours,))
            
            history = []
            for row in cur.fetchall():
                history.append({
                    'timestamp': row[0].isoformat(),
                    'price': row[1],
                    'portfolio_value': row[2],
                    'roi': row[3]
                })
            return history
        