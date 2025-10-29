# Solana Price Predictor

Prédiction du prix du Solana en utilisant un modèle Transformer avec deep learning.

## Architecture

- **Modèle** : Transformer (8 couches, 512 dimensions, 16 têtes d'attention)
- **Features** : 20 indicateurs techniques (prix, volume, RSI, MACD, moyennes mobiles, Bollinger)
- **Horizons de prédiction** : 1h, 6h, 24h
- **Entraînement** : RTX 4070, ~10M paramètres

## Structure du projet

Solana-predictor/
├── .venv/              # Environnement virtuel (non versionné)
├── data/               # Données CSV et graphiques (non versionné)
├── models/             # Modèles entraînés (non versionné)
├── logs/               # Logs du paper trading (non versionné)
├── src/
│   ├── fetch_data.py          # Récupération données Yahoo Finance
│   ├── prepare_data.py        # Préparation et indicateurs techniques
│   ├── train_transformer.py   # Entraînement du modèle
│   ├── backtest_final.py      # Backtesting avec stratégie optimisée
│   ├── optimize_thresholds.py # Optimisation des paramètres
│   └── update_data.py         # Mise à jour des données
├── app.py              # Interface Streamlit
├── paper_trading.py    # Bot de paper trading temps réel
├── test_gpu.py         # Test GPU
├── requirements.txt
└── README.md

## Installation

### 1. Cloner le repo
```bash
git clone https://github.com/TON_USERNAME/Solana-predictor.git
cd Solana-predictor

2. Créer l'environnement virtuel

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

3. Installer les dépendances

pip install -r requirements.txt

4. Vérifier que PyTorch reconnaît ta GPU
python test_gpu.py

Récupérer les données

python src/fetch_data.py
python src/prepare_data.py

Entraîner le modèle
python src/train_transformer.py

Backtesting
python src/backtest_final.py

Optimiser les paramètres
python src/optimize_thresholds.py

Interface web
streamlit run app.py

Paper trading (mode simulation)
python paper_trading.py

Résultats du backtesting
Période de test : Sept 2024 - Oct 2025 (données non vues pendant l'entraînement)

ROI : +194.55%
Alpha : +126.47% (vs Buy&Hold à +68%)
Sharpe Ratio : 3.67
Max Drawdown : -24.91%
Trades : 62 sur 1 an
Win Rate : 38.5%

Paramètres optimisés

Seuil d'achat : +1.5%
Seuil de vente : -1.5%
Stop Loss : -12%
Take Profit : +25%
Taille de position : 30%
Frais : 0.3% par trade

Avertissements
⚠️ Ce projet est strictement éducatif

Les performances passées ne garantissent pas les résultats futurs
Les cryptomonnaies sont extrêmement volatiles
Ne jamais utiliser avec de l'argent réel sans tests approfondis
Le backtesting peut surestimer les performances (slippage, liquidité, etc.)
Recommandation : 3+ mois de paper trading profitable avant tout usage réel

Technologies

Python 3.10+
PyTorch 2.0+ (GPU requis pour entraînement)
Streamlit (interface)
Yahoo Finance API (données gratuites)
scikit-learn (prétraitement)
Plotly (visualisations) 