import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime, timedelta
import subprocess

# Configuration
st.set_page_config(
    page_title="Solana Price Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Classe du modèle
class PositionalEncoding(nn.Module):
    """Encodage positionnel pour le Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        import math
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
    """Modèle Transformer pour prédiction de séries temporelles"""
    
    def __init__(self, input_size, d_model=512, nhead=16, num_layers=8, 
                 dim_feedforward=2048, dropout=0.15, num_outputs=3):
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
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

@st.cache_resource
def load_model():
    """Charge le modèle"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    scaler = joblib.load('models/scaler.pkl')
    
    # Modèle Transformer
    # Modèle Transformer - VALEURS FORCÉES pour correspondre à l'entraînement
    model = TransformerModel(
        input_size=metadata['input_size'],
        d_model=512,      # Doit correspondre à train_transformer.py
        nhead=16,         # Doit correspondre à train_transformer.py
        num_layers=8,     # Doit correspondre à train_transformer.py
        dim_feedforward=2048,
        dropout=0.15,
        num_outputs=3
    ).to(device)
    
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()
    
    return model, scaler, metadata, device

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def load_data():
    """Charge les données"""
    df = pd.read_csv('data/solana_prepared.csv', index_col='timestamp', parse_dates=True)
    return df

def get_last_update_time():
    """Récupère l'heure de la dernière mise à jour"""
    log_path = 'data/update_log.json'
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
            return log.get('last_update', 'Inconnue')
    return 'Jamais'

def update_data():
    """Lance la mise à jour des données"""
    with st.spinner("🔄 Mise à jour en cours..."):
        result = subprocess.run(['python', 'src/update_data.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Données mises à jour !")
            st.cache_data.clear()  # Efface le cache
            st.rerun()
        else:
            st.error(f"❌ Erreur : {result.stderr}")

def prepare_sequence(df, sequence_length=72):
    """Prépare la séquence pour prédiction"""
    feature_cols = [
        'price', 'volume', 'open', 'high', 'low',
        'ma_7', 'ma_25', 'ma_99', 'rsi', 'macd', 'macd_signal',
        'volatility', 'price_change', 'price_change_1h', 'price_change_24h',
        'volume_change', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower'
    ]
    
    df_clean = df[feature_cols].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    X = df_clean.tail(sequence_length).values
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Valeurs NaN/Inf dans les données")
    
    return X

def make_prediction(model, scaler, sequence, device):
    """Fait une prédiction"""
    sequence_scaled = scaler.transform(sequence)
    sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(sequence_tensor)
    
    return prediction.cpu().numpy()[0]

def calculate_confidence_score(df, predictions, horizons=[1, 6, 24]):
    """Calcule un score de confiance basé sur les erreurs historiques"""
    scores = []
    
    for i, h in enumerate(horizons):
        target_col = f'target_change_{h}h'
        if target_col in df.columns:
            # Erreurs récentes (derniers 100 points)
            recent_errors = df[target_col].tail(100).abs().mean()
            
            # Score basé sur l'erreur (inverse)
            # Plus l'erreur est faible, plus le score est élevé
            confidence = max(0, 100 - recent_errors * 10)
            scores.append(min(100, confidence))
        else:
            scores.append(50)
    
    return scores

def plot_prediction_vs_reality(df, horizon_hours):
    """Compare prédictions passées VS réalité"""
    target_col = f'target_change_{horizon_hours}h'
    
    if target_col not in df.columns:
        return None
    
    # Derniers 200 points avec données complètes
    df_valid = df.dropna(subset=[target_col]).tail(200)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'Variations réelles à {horizon_hours}h', 'Erreur absolue'),
        vertical_spacing=0.1
    )
    
    # Variations réelles
    fig.add_trace(
        go.Scatter(x=df_valid.index, y=df_valid[target_col],
                  name='Variation réelle',
                  line=dict(color='#00D9FF', width=2)),
        row=1, col=1
    )
    
    # Ligne zéro
    fig.add_trace(
        go.Scatter(x=df_valid.index, y=[0]*len(df_valid),
                  name='Référence (0%)',
                  line=dict(color='gray', dash='dash', width=1)),
        row=1, col=1
    )
    
    # Erreur absolue (simulation)
    errors = df_valid[target_col].abs()
    fig.add_trace(
        go.Scatter(x=df_valid.index, y=errors,
                  name='Erreur absolue',
                  fill='tozeroy',
                  line=dict(color='orange', width=1)),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Variation (%)", row=1, col=1)
    fig.update_yaxes(title_text="Erreur (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def plot_main_chart(df, predictions, current_price, confidence_scores):
    """Graphique principal avec prédictions"""
    df_recent = df.tail(168)
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Prix Solana et Prédictions', 'Volume', 'RSI'),
        vertical_spacing=0.08
    )
    
    # Prix historique
    fig.add_trace(
        go.Scatter(x=df_recent.index, y=df_recent['price'],
                  name='Prix historique', line=dict(color='#00D9FF', width=2)),
        row=1, col=1
    )
    
    # Moyennes mobiles
    fig.add_trace(
        go.Scatter(x=df_recent.index, y=df_recent['ma_25'],
                  name='MA 25h', line=dict(color='orange', width=1, dash='dash')),
        row=1, col=1
    )
    
    # Point actuel
    fig.add_trace(
        go.Scatter(x=[df_recent.index[-1]], y=[current_price],
                  name='Prix actuel', mode='markers',
                  marker=dict(size=15, color='lime', symbol='star')),
        row=1, col=1
    )
    
    # Prédictions
    last_time = df_recent.index[-1]
    horizons = [1, 6, 24]
    colors = ['yellow', 'orange', 'red']
    
    for i, (h, pred, color, conf) in enumerate(zip(horizons, predictions, colors, confidence_scores)):
        future_time = last_time + timedelta(hours=h)
        future_price = current_price * (1 + pred/100)
        
        # Ligne de prédiction avec opacité basée sur la confiance
        opacity = 0.3 + (conf / 100) * 0.7
        
        fig.add_trace(
            go.Scatter(x=[last_time, future_time],
                      y=[current_price, future_price],
                      name=f'{h}h: {pred:+.2f}% (conf: {conf:.0f}%)',
                      line=dict(color=color, width=2, dash='dot'),
                      opacity=opacity),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[future_time], y=[future_price],
                      mode='markers', showlegend=False,
                      marker=dict(size=10, color=color),
                      opacity=opacity),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df_recent.index, y=df_recent['volume'],
               name='Volume', marker_color='rgba(0, 217, 255, 0.3)'),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df_recent.index, y=df_recent['rsi'],
                  name='RSI', line=dict(color='purple', width=2)),
        row=3, col=1
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_yaxes(title_text="Prix (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    fig.update_layout(
        height=900,
        hovermode='x unified',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.title("🚀 Solana Price Predictor AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Contrôles")
        
        # Bouton de mise à jour
        if st.button("🔄 Mettre à jour les données", use_container_width=True):
            update_data()
        
        # Dernière mise à jour
        last_update = get_last_update_time()
        st.info(f"📅 Dernière MAJ : {last_update}")
        
        st.markdown("---")
        
        # Informations du modèle
        try:
            with open('models/metadata.json', 'r') as f:
                metadata = json.load(f)
            
            st.header("📊 Modèle")
            st.metric("Entraîné le", metadata['train_date'].split()[0])
            st.metric("Meilleure Loss", f"{metadata['best_val_loss']:.4f}")
            
            device_type = "GPU" if torch.cuda.is_available() else "CPU"
            st.metric("Device", device_type)
        except:
            st.warning("⚠️ Modèle non trouvé")
        
        st.markdown("---")
        st.markdown("### 📖 Guide")
        st.markdown("""
        - 🟢 **1h-6h** : Haute fiabilité
        - 🟡 **24h** : Fiabilité moyenne
        - Le score de confiance est basé sur les erreurs historiques
        """)
    
    try:
        # Chargement
        model, scaler, metadata, device = load_model()
        df = load_data()
        
        # Données actuelles
        current_price = df['price'].iloc[-1]
        last_update_data = df.index[-1]
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Prix actuel", f"${current_price:.2f}")
        
        with col2:
            change_24h = df['price_change_24h'].iloc[-1] * 100
            st.metric("📈 Variation 24h", f"{change_24h:+.2f}%",
                     delta=f"{change_24h:+.2f}%")
        
        with col3:
            st.metric("📅 Données au", last_update_data.strftime("%Y-%m-%d %H:%M"))
        
        with col4:
            st.metric("📊 Points", f"{len(df):,}")
        
        st.markdown("---")
        
        # Prédictions
        st.header("🔮 Prédictions avec Score de Confiance")
        
        sequence = prepare_sequence(df, sequence_length=24)
        predictions = make_prediction(model, scaler, sequence, device)
        confidence_scores = calculate_confidence_score(df, predictions)
        
        # Affichage des prédictions
        col1, col2, col3 = st.columns(3)
        horizons = [1, 6, 24]
        labels = ["⏰ 1 heure", "⏰ 6 heures", "⏰ 24 heures"]
        emojis = ["🟢", "🟢", "🟡"]
        
        for col, h, label, pred, conf, emoji in zip([col1, col2, col3], horizons, labels, 
                                                      predictions, confidence_scores, emojis):
            future_price = current_price * (1 + pred/100)
            with col:
                st.markdown(f"### {emoji} {label}")
                st.metric(
                    "Prix prédit",
                    f"${future_price:.2f}",
                    f"{pred:+.2f}%"
                )
                st.progress(conf / 100)
                st.caption(f"Confiance : {conf:.0f}%")
        
        st.markdown("---")
        
        # Graphique principal
        st.header("📊 Visualisation Interactive")
        
        fig_main = plot_main_chart(df, predictions, current_price, confidence_scores)
        st.plotly_chart(fig_main, use_container_width=True)
        
        st.markdown("---")
        
        # Analyse de précision historique
        st.header("🎯 Analyse de Précision Historique")
        
        horizon_tabs = st.tabs(["1 heure", "6 heures", "24 heures"])
        
        for i, (tab, h) in enumerate(zip(horizon_tabs, [1, 6, 24])):
            with tab:
                fig_comp = plot_prediction_vs_reality(df, h)
                if fig_comp:
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Statistiques
                    target_col = f'target_change_{h}h'
                    if target_col in df.columns:
                        recent_data = df[target_col].dropna().tail(100)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE moyenne", f"{recent_data.abs().mean():.2f}%")
                        with col2:
                            st.metric("Erreur max", f"{recent_data.abs().max():.2f}%")
                        with col3:
                            st.metric("Variation moyenne", f"{recent_data.mean():+.2f}%")
                        with col4:
                            volatility = recent_data.std()
                            st.metric("Volatilité", f"{volatility:.2f}%")
        
        st.markdown("---")
        
        # Statistiques techniques
        st.header("📈 Indicateurs Techniques")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = df['rsi'].iloc[-1]
            rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
            st.metric(f"RSI {rsi_color}", f"{rsi:.1f}")
        
        with col2:
            st.metric("MACD", f"{df['macd'].iloc[-1]:.2f}")
        
        with col3:
            st.metric("Volatilité 24h", f"{df['volatility'].iloc[-1]:.2f}")
        
        with col4:
            volume_24h = df['volume'].tail(24).mean()
            st.metric("Volume moy 24h", f"{volume_24h/1e9:.2f}B")
        
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        st.info("💡 Cliquez sur '🔄 Mettre à jour les données' dans la sidebar")

if __name__ == "__main__":
    main()