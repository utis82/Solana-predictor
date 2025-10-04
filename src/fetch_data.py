import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_solana_data():
    """
    Récupère les données historiques de Solana depuis Yahoo Finance
    """
    print("=" * 60)
    print("🚀 RÉCUPÉRATION DES DONNÉES SOLANA")
    print("=" * 60)
    print("📥 Téléchargement depuis Yahoo Finance...")
    
    try:
        # SOL-USD = Solana sur Yahoo Finance
        # interval='1h' = données horaires
        # period='max' = maximum disponible
        sol = yf.Ticker("SOL-USD")
        
        # Récupération données horaires (max ~730 jours = 2 ans)
        print("⏳ Récupération des données horaires (2 ans max)...")
        df_hourly = sol.history(period='730d', interval='1h')
        
        if df_hourly.empty:
            print("❌ Aucune donnée horaire disponible")
            return None
        
        # Nettoyage et renommage
        df_hourly = df_hourly.reset_index()
        df_hourly = df_hourly.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'price',
            'Volume': 'volume'
        })
        
        # Sélection des colonnes importantes
        df_hourly = df_hourly[['timestamp', 'open', 'high', 'low', 'price', 'volume']]
        df_hourly.set_index('timestamp', inplace=True)
        
        # Suppression des valeurs manquantes
        df_hourly = df_hourly.dropna()
        
        print(f"✅ Données horaires récupérées : {len(df_hourly)} points")
        print(f"📅 Période : {df_hourly.index[0]} à {df_hourly.index[-1]}")
        
        # Récupération données journalières (max disponible)
        print("\n⏳ Récupération des données journalières (historique complet)...")
        df_daily = sol.history(period='max', interval='1d')
        
        if not df_daily.empty:
            df_daily = df_daily.reset_index()
            df_daily = df_daily.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'price',
                'Volume': 'volume'
            })
            df_daily = df_daily[['timestamp', 'open', 'high', 'low', 'price', 'volume']]
            df_daily.set_index('timestamp', inplace=True)
            df_daily = df_daily.dropna()
            
            print(f"✅ Données journalières récupérées : {len(df_daily)} points")
            print(f"📅 Période : {df_daily.index[0]} à {df_daily.index[-1]}")
        
        return df_hourly, df_daily
    
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None, None

def save_data(df, filename):
    """Sauvegarde les données"""
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    df.to_csv(filepath)
    print(f"💾 Sauvegardé : {filepath}")
    print(f"📦 Taille : {os.path.getsize(filepath) / 1024:.2f} KB")

def display_stats(df, name):
    """Affiche les statistiques"""
    print(f"\n📊 Statistiques {name} :")
    print(f"  • Nombre de points : {len(df)}")
    print(f"  • Prix min : ${df['price'].min():.2f}")
    print(f"  • Prix max : ${df['price'].max():.2f}")
    print(f"  • Prix moyen : ${df['price'].mean():.2f}")
    print(f"  • Prix actuel : ${df['price'].iloc[-1]:.2f}")
    print(f"  • Valeurs manquantes : {df.isnull().sum().sum()}")
    
    print(f"\n📈 Aperçu des dernières données :")
    print(df.tail())

def main():
    # Récupération
    result = fetch_solana_data()
    
    if result is None:
        print("\n❌ Échec complet de la récupération")
        return
    
    df_hourly, df_daily = result
    
    # Sauvegarde et stats pour données horaires
    if df_hourly is not None and not df_hourly.empty:
        print("\n" + "=" * 60)
        save_data(df_hourly, 'solana_hourly.csv')
        display_stats(df_hourly, "HORAIRES")
    
    # Sauvegarde et stats pour données journalières
    if df_daily is not None and not df_daily.empty:
        print("\n" + "=" * 60)
        save_data(df_daily, 'solana_daily.csv')
        display_stats(df_daily, "JOURNALIÈRES")
    
    print("\n" + "=" * 60)
    print("✅ DONNÉES PRÊTES POUR L'ENTRAÎNEMENT !")
    print("=" * 60)
    print("\n💡 Prochaine étape : Préparation des données pour le modèle")

if __name__ == "__main__":
    main()