import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(filename='solana_hourly.csv'):
    """Charge les données"""
    filepath = os.path.join('data', filename)
    print(f"📂 Chargement : {filepath}")
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    print(f"✅ {len(df)} points chargés")
    return df

def add_technical_indicators(df):
    """Ajoute des indicateurs techniques"""
    print("\n🔧 Ajout des indicateurs techniques...")
    
    df = df.copy()
    
    # 1. Moyennes mobiles
    df['ma_7'] = df['price'].rolling(window=7).mean()
    df['ma_25'] = df['price'].rolling(window=25).mean()
    df['ma_99'] = df['price'].rolling(window=99).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    exp1 = df['price'].ewm(span=12, adjust=False).mean()
    exp2 = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 4. Volatilité
    df['volatility'] = df['price'].rolling(window=24).std()
    
    # 5. Variations de prix
    df['price_change'] = df['price'].pct_change()
    df['price_change_1h'] = df['price'].pct_change(periods=1)
    df['price_change_24h'] = df['price'].pct_change(periods=24)
    
    # 6. Volume variation
    df['volume_change'] = df['volume'].pct_change()
    
    # 7. Bandes de Bollinger
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    df['bb_std'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Suppression des NaN créés par les indicateurs
    df = df.dropna()
    
    print(f"✅ Indicateurs ajoutés. {len(df)} points restants après nettoyage")
    print(f"\n📊 Nouvelles colonnes : {list(df.columns)}")
    
    return df

def create_targets(df, horizons=[1, 6, 24, 168]):
    """
    Crée les cibles de prédiction pour différents horizons
    horizons : liste d'heures dans le futur
    1h, 6h, 24h (1 jour), 168h (7 jours)
    """
    print(f"\n🎯 Création des cibles pour horizons : {horizons} heures...")
    
    df = df.copy()
    
    for h in horizons:
        # Prix futur
        df[f'target_price_{h}h'] = df['price'].shift(-h)
        
        # Variation en pourcentage
        df[f'target_change_{h}h'] = ((df[f'target_price_{h}h'] - df['price']) / df['price']) * 100
        
        # Direction (monte/descend)
        df[f'target_direction_{h}h'] = (df[f'target_change_{h}h'] > 0).astype(int)
    
    # Suppression des dernières lignes sans cible
    max_horizon = max(horizons)
    df = df.iloc[:-max_horizon]
    
    print(f"✅ Cibles créées. {len(df)} points utilisables pour l'entraînement")
    
    return df

def visualize_data(df, save_path='data/visualizations'):
    """Crée des visualisations"""
    print("\n📊 Création des visualisations...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Style
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Prix et moyennes mobiles
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Graphique 1 : Prix et MA
    ax1 = axes[0]
    ax1.plot(df.index, df['price'], label='Prix', linewidth=1, alpha=0.7)
    ax1.plot(df.index, df['ma_7'], label='MA 7h', linewidth=1.5)
    ax1.plot(df.index, df['ma_25'], label='MA 25h', linewidth=1.5)
    ax1.plot(df.index, df['ma_99'], label='MA 99h', linewidth=1.5)
    ax1.set_title('Prix Solana et Moyennes Mobiles', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Prix (USD)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2 : RSI
    ax2 = axes[1]
    ax2.plot(df.index, df['rsi'], color='purple', linewidth=1)
    ax2.axhline(y=70, color='r', linestyle='--', label='Sur-acheté (70)')
    ax2.axhline(y=30, color='g', linestyle='--', label='Sur-vendu (30)')
    ax2.set_title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Graphique 3 : Volume
    ax3 = axes[2]
    ax3.bar(df.index, df['volume'], alpha=0.5, color='blue')
    ax3.set_title('Volume de transactions', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Volume', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'price_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Graphique 1 sauvegardé : {save_path}/price_analysis.png")
    plt.close()
    
    # 2. Distribution des variations de prix
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, h in enumerate([1, 6, 24]):
        ax = axes[idx // 2, idx % 2]
        df[f'target_change_{h}h'].hist(bins=50, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution des variations à {h}h', fontsize=12, fontweight='bold')
        ax.set_xlabel('Variation (%)', fontsize=10)
        ax.set_ylabel('Fréquence', fontsize=10)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'price_changes_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Graphique 2 sauvegardé : {save_path}/price_changes_distribution.png")
    plt.close()
    
    # 3. Matrice de corrélation
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_cols = ['price', 'volume', 'ma_7', 'ma_25', 'rsi', 'macd', 'volatility']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Corrélation'})
    ax.set_title('Matrice de corrélation des features', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Graphique 3 sauvegardé : {save_path}/correlation_matrix.png")
    plt.close()
    
    print(f"\n✅ Toutes les visualisations sauvegardées dans : {save_path}/")

def save_prepared_data(df, filename='solana_prepared.csv'):
    """Sauvegarde les données préparées"""
    filepath = os.path.join('data', filename)
    df.to_csv(filepath)
    print(f"\n💾 Données préparées sauvegardées : {filepath}")
    print(f"📦 Taille : {os.path.getsize(filepath) / 1024:.2f} KB")
    print(f"📊 Shape : {df.shape}")

def main():
    print("=" * 70)
    print("🔧 PRÉPARATION DES DONNÉES POUR L'ENTRAÎNEMENT")
    print("=" * 70)
    
    # 1. Chargement
    df = load_data('solana_hourly.csv')
    
    # 2. Ajout indicateurs techniques
    df = add_technical_indicators(df)
    
    # 3. Création des cibles
    df = create_targets(df, horizons=[1, 6, 24])
    
    # 4. Visualisations
    visualize_data(df)
    
    # 5. Sauvegarde
    save_prepared_data(df)
    
    # 6. Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES DONNÉES PRÉPARÉES")
    print("=" * 70)
    print(f"Total de points : {len(df)}")
    print(f"Période : {df.index[0]} à {df.index[-1]}")
    print(f"Nombre de features : {len([c for c in df.columns if not c.startswith('target')])}")
    print(f"Nombre de cibles : {len([c for c in df.columns if c.startswith('target')])}")
    print("\n✅ DONNÉES PRÊTES POUR L'ENTRAÎNEMENT DU MODÈLE !")
    print("\n💡 Prochaine étape : Création et entraînement du modèle LSTM")

if __name__ == "__main__":
    main()