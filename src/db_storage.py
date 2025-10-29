import os
import json
import psycopg2
from datetime import datetime
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse

class DatabaseStorage:
    def __init__(self):
        database_url = os.environ.get('DATABASE_URL')

        if not database_url:
            raise ValueError("DATABASE_URL doit être définie")
        
        # Parser l'URL
        result = urlparse(database_url)
        
        # Construire les paramètres de connexion explicitement
        conn_params = {
            'dbname': result.path[1:],  # Enlever le / initial
            'user': result.username,
            'password': result.password,
            'host': result.hostname,
            'port': result.port or 5432,
            'sslmode': 'require'
        }
        
        print(f"Connexion à {conn_params['host']}:{conn_params['port']}...")
        
        # CRÉER LA CONNEXION D'ABORD
        self.conn = psycopg2.connect(**conn_params)
        self.conn.autocommit = True
        print("Connexion réussie!")
        
        # ENSUITE initialiser les tables
        self._init_tables()

    def _init_tables(self):
        """Initialise les tables si elles n'existent pas"""
        with self.conn.cursor() as cur:
            # Créer la table bot_state
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    capital DECIMAL(20, 8) DEFAULT 1000.0,
                    position VARCHAR(10) DEFAULT 'none',
                    entry_price DECIMAL(20, 8) DEFAULT 0,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT single_row_check CHECK (id = 1)
                )
            """)
            
            # Insérer la ligne initiale si elle n'existe pas
            cur.execute("""
                INSERT INTO bot_state (id, capital, position, entry_price) 
                VALUES (1, 1000.0, 'none', 0.0)
                ON CONFLICT (id) DO NOTHING
            """)
            
            # Créer la table trades avec toutes les colonnes nécessaires
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action VARCHAR(10),
                    price DECIMAL(20, 8),
                    amount DECIMAL(20, 8),
                    capital DECIMAL(20, 8),
                    predicted_change DECIMAL(10, 4),
                    rsi DECIMAL(10, 4),
                    fee DECIMAL(20, 8),
                    pnl DECIMAL(20, 8),
                    pnl_pct DECIMAL(10, 4)
                )
            """)
            
            # Créer la table portfolio_history
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    price DECIMAL(20, 8),
                    portfolio_value DECIMAL(20, 8),
                    roi DECIMAL(10, 4)
                )
            """)
            
            self.conn.commit()
            print("Tables initialisées avec succès!")
    
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
                    'capital': float(result[0]),
                    'position': result[1],
                    'entry_price': float(result[2])
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
                    'price': float(row[2]),
                    'amount': float(row[3]),
                    'capital': float(row[4]),
                    'predicted_change': float(row[5]),
                    'rsi': float(row[6]),
                    'fee': float(row[7]),
                    'pnl': float(row[8]),
                    'pnl_pct': float(row[9])
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
                    'price': float(row[1]),
                    'portfolio_value': float(row[2]),
                    'roi': float(row[3])
                })
            return history