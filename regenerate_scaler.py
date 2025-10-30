import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("📂 Chargement des données...")
df = pd.read_csv('data/solana_prepared.csv', index_col='timestamp', parse_dates=True)

feature_cols = [
    'price', 'volume', 'open', 'high', 'low',
    'ma_7', 'ma_25', 'ma_99',
    'rsi', 'macd', 'macd_signal',
    'volatility', 'price_change', 'price_change_1h', 'price_change_24h',
    'volume_change', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower'
]

X = df[feature_cols].values

print("🧹 Nettoyage...")
X = np.where(np.isinf(X), np.nan, X)
nan_rows = np.any(np.isnan(X), axis=1)
X = X[~nan_rows]

print("🔄 Création du nouveau scaler...")
scaler = StandardScaler()
scaler.fit(X)

print("💾 Sauvegarde avec protocol=4...")
joblib.dump(scaler, 'models/scaler.pkl', protocol=4)

print("✅ Nouveau scaler créé et sauvegardé !")

# Test de chargement
test = joblib.load('models/scaler.pkl')
print("✅ Vérification OK !")