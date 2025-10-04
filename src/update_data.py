import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

def fetch_latest_data():
    """Récupère les toutes dernières données de Solana"""
    print(f"📥 Récupération des dernières données Solana...")
    
    try:
        sol = yf.Ticker("SOL-USD")
        
        # Données horaires récentes (7 jours)
        df_hourly = sol.history(period='7d', interval='1h')
        
        if df_hourly.empty:
            print("❌ Aucune donnée disponible")
            return None
        
        # Nettoyage
        df_hourly = df_hourly.reset_index()
        df_hourly = df_hourly.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'price',
            'Volume': 'volume'
        })
        
        df_hourly = df_hourly[['timestamp', 'open', 'high', 'low', 'price', 'volume']]
        df_hourly.set_index('timestamp', inplace=True)
        df_hourly = df_hourly.dropna()
        
        print(f"✅ {len(df_hourly)} points récupérés")
        print(f"📅 Dernière donnée : {df_hourly.index[-1]}")
        print(f"💰 Prix actuel : ${df_hourly['price'].iloc[-1]:.2f}")
        
        return df_hourly
    
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

def add_technical_indicators(df):
    """Ajoute les indicateurs techniques"""
    df = df.copy()
    
    # Moyennes mobiles
    df['ma_7'] = df['price'].rolling(window=7).mean()
    df['ma_25'] = df['price'].rolling(window=25).mean()
    df['ma_99'] = df['price'].rolling(window=99).mean()
    
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['price'].ewm(span=12, adjust=False).mean()
    exp2 = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Volatilité
    df['volatility'] = df['price'].rolling(window=24).std()
    
    # Variations
    df['price_change'] = df['price'].pct_change()
    df['price_change_1h'] = df['price'].pct_change(periods=1)
    df['price_change_24h'] = df['price'].pct_change(periods=24)
    df['volume_change'] = df['volume'].pct_change()
    
    # Bandes de Bollinger
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    df['bb_std'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Nettoyage
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def merge_with_existing_data(new_df):
    """Fusionne avec les données existantes"""
    filepath = 'data/solana_prepared.csv'
    
    if os.path.exists(filepath):
        print(f"\n📂 Chargement des données existantes...")
        existing_df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        
        # Concaténation
        combined_df = pd.concat([existing_df, new_df])
        
        # Suppression des doublons (garder le plus récent)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # Tri chronologique
        combined_df = combined_df.sort_index()
        
        print(f"✅ Fusion réussie : {len(existing_df)} → {len(combined_df)} points")
        
        return combined_df
    else:
        print(f"⚠️  Pas de données existantes, création d'un nouveau fichier")
        return new_df

def save_update_log():
    """Sauvegarde l'heure de la dernière mise à jour"""
    log = {
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'success'
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/update_log.json', 'w') as f:
        json.dump(log, f, indent=4)

def main():
    print("=" * 70)
    print("🔄 MISE À JOUR DES DONNÉES SOLANA")
    print("=" * 70)
    
    # 1. Récupération
    df_new = fetch_latest_data()
    
    if df_new is None:
        print("\n❌ Échec de la mise à jour")
        return
    
    # 2. Ajout indicateurs
    print("\n🔧 Ajout des indicateurs techniques...")
    df_new = add_technical_indicators(df_new)
    print(f"✅ {len(df_new)} points après nettoyage")
    
    # 3. Fusion avec données existantes
    df_combined = merge_with_existing_data(df_new)
    
    # 4. Sauvegarde
    filepath = 'data/solana_prepared.csv'
    df_combined.to_csv(filepath)
    print(f"\n💾 Données sauvegardées : {filepath}")
    
    # 5. Log
    save_update_log()
    
    print("\n" + "=" * 70)
    print("✅ MISE À JOUR TERMINÉE !")
    print("=" * 70)
    print(f"📊 Total de points : {len(df_combined)}")
    print(f"📅 Dernière donnée : {df_combined.index[-1]}")
    print(f"💰 Prix actuel : ${df_combined['price'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()