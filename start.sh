#!/bin/bash

# Télécharger le modèle depuis Hugging Face
echo "📥 Téléchargement du modèle..."
python download_model.py

# Lancer le bot de trading en arrière-plan
echo "🤖 Démarrage du bot de trading..."
python railway_bot.py &

# Lancer le dashboard Streamlit
echo "🎨 Démarrage du dashboard..."
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
