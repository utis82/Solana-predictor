import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import math

# Vérification GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device utilisé : {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU : {torch.cuda.get_device_name(0)}")
    print(f"💾 Mémoire GPU disponible : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

class SolanaDataset(Dataset):
    """Dataset pour les séquences temporelles"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    """Encodage positionnel pour le Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Création de l'encodage positionnel
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
    
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, num_outputs=3):
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Projection de l'input vers d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Encodage positionnel
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU est meilleur que ReLU pour les Transformers
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Couches de sortie
        self.fc1 = nn.Linear(d_model, 128)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_outputs)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # Projection vers d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Ajout de l'encodage positionnel
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global Average Pooling sur la dimension temporelle
        x = torch.mean(x, dim=1)  # (batch, d_model)
        
        # Layer Normalization
        x = self.layer_norm(x)
        
        # Couches fully connected
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

def load_and_prepare_data(filepath='data/solana_prepared.csv', sequence_length=48):
    """Charge et prépare les données en séquences"""
    print(f"\n📂 Chargement des données : {filepath}")
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    print(f"✅ {len(df)} points chargés")
    
    # Sélection des features
    feature_cols = [
        'price', 'volume', 'open', 'high', 'low',
        'ma_7', 'ma_25', 'ma_99',
        'rsi', 'macd', 'macd_signal',
        'volatility', 'price_change', 'price_change_1h', 'price_change_24h',
        'volume_change', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower'
    ]
    
    # Cibles
    target_cols = ['target_change_1h', 'target_change_6h', 'target_change_24h']
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    print(f"\n📊 Features shape avant nettoyage : {X.shape}")
    print(f"📊 Targets shape avant nettoyage : {y.shape}")
    
    # Nettoyage
    print(f"\n🧹 Nettoyage des valeurs problématiques...")
    
    X = np.where(np.isinf(X), np.nan, X)
    y = np.where(np.isinf(y), np.nan, y)
    
    nan_rows_X = np.any(np.isnan(X), axis=1)
    nan_rows_y = np.any(np.isnan(y), axis=1)
    nan_rows = nan_rows_X | nan_rows_y
    
    print(f"⚠️  Lignes avec NaN/Inf : {nan_rows.sum()}")
    
    X = X[~nan_rows]
    y = y[~nan_rows]
    
    print(f"✅ Données nettoyées : X={X.shape}, y={y.shape}")
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("❌ Il reste des NaN/Inf dans X !")
        return None, None, None, None, None
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("❌ Il reste des NaN/Inf dans y !")
        return None, None, None, None, None
    
    # Normalisation
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Création des séquences (plus longues pour le Transformer)
    print(f"\n🔄 Création des séquences (longueur = {sequence_length})...")
    X_seq, y_seq = [], []
    
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"✅ Séquences créées : X={X_seq.shape}, y={y_seq.shape}")
    
    return X_seq, y_seq, scaler_X, feature_cols, target_cols

def split_data(X, y, train_size=0.7, val_size=0.15):
    """Split train/val/test de manière temporelle"""
    print(f"\n✂️  Split des données : {train_size*100:.0f}% train, {val_size*100:.0f}% val, {(1-train_size-val_size)*100:.0f}% test")
    
    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"✅ Train : {len(X_train)} samples")
    print(f"✅ Val   : {len(X_val)} samples")
    print(f"✅ Test  : {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.0001):
    """Entraîne le modèle"""
    print(f"\n🚀 Début de l'entraînement...")
    print(f"⚙️  Epochs : {num_epochs}, Learning rate : {lr}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Scheduler pour réduire le learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping pour stabiliser l'entraînement
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1
        
        # Affichage
        if (epoch + 1) % 5 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping à l'epoch {epoch+1}")
            break
    
    print(f"\n✅ Entraînement terminé !")
    print(f"🏆 Meilleure val loss : {best_val_loss:.6f}")
    
    return train_losses, val_losses

def plot_training(train_losses, val_losses):
    """Affiche les courbes d'apprentissage"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Courbes d\'apprentissage - Transformer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.title('Courbes d\'apprentissage (échelle log)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Courbes sauvegardées : models/training_curves.png")
    plt.close()

def evaluate_model(model, test_loader, target_cols):
    """Évalue le modèle sur les données de test"""
    print(f"\n🧪 Évaluation sur les données de test...")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calcul des métriques
    print("\n📊 Performances par horizon :")
    for i, col in enumerate(target_cols):
        mae = np.mean(np.abs(all_preds[:, i] - all_targets[:, i]))
        rmse = np.sqrt(np.mean((all_preds[:, i] - all_targets[:, i])**2))
        mape = np.mean(np.abs((all_preds[:, i] - all_targets[:, i]) / (all_targets[:, i] + 1e-8))) * 100
        print(f"  {col:20s} - MAE: {mae:.4f}%, RMSE: {rmse:.4f}%, MAPE: {mape:.4f}%")
    
    return all_preds, all_targets

def save_model_info(model, scaler_X, feature_cols, target_cols, train_losses, val_losses):
    """Sauvegarde les informations du modèle"""
    os.makedirs('models', exist_ok=True)
    
    import joblib
    joblib.dump(scaler_X, 'models/scaler.pkl')
    
    metadata = {
        'model_type': 'Transformer',
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'input_size': len(feature_cols),
        'num_outputs': len(target_cols),
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'best_val_loss': float(min(val_losses))
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n💾 Modèle et métadonnées sauvegardés dans : models/")

def main():
    print("=" * 70)
    print("🧠 ENTRAÎNEMENT DU MODÈLE TRANSFORMER")
    print("=" * 70)
    
    # Paramètres optimisés pour ta 4070 - VERSION HARDCORE
    SEQUENCE_LENGTH = 72  # Encore plus long (3 jours de contexte)
    BATCH_SIZE = 256  # Batch énorme pour saturer la GPU
    NUM_EPOCHS = 300  # Plus d'epochs
    LEARNING_RATE = 0.0001
    
    # 1. Chargement et préparation
    X, y, scaler_X, feature_cols, target_cols = load_and_prepare_data(sequence_length=SEQUENCE_LENGTH)
    
    if X is None or y is None:
        print("\n❌ Erreur lors de la préparation des données. Arrêt.")
        return
    
    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 3. Datasets et DataLoaders
    train_dataset = SolanaDataset(X_train, y_train)
    val_dataset = SolanaDataset(X_val, y_val)
    test_dataset = SolanaDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    
    # 4. Création du modèle Transformer - VERSION HARDCORE
    model = TransformerModel(
        input_size=len(feature_cols),
        d_model=512,         # Doublé ! (256 → 512)
        nhead=16,            # Doublé ! (8 → 16 têtes d'attention)
        num_layers=8,        # Doublé ! (4 → 8 couches)
        dim_feedforward=2048, # Quadruplé ! (512 → 2048)
        dropout=0.15,
        num_outputs=len(target_cols)
    ).to(device)
    
    print(f"\n🏗️  Architecture du modèle Transformer :")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Nombre total de paramètres : {total_params:,}")
    print(f"📊 Paramètres entraînables : {trainable_params:,}")
    print(f"💾 Taille du modèle : {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 5. Entraînement
    train_losses, val_losses = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # 6. Chargement du meilleur modèle
    model.load_state_dict(torch.load('models/best_model.pth'))
    
    # 7. Évaluation
    all_preds, all_targets = evaluate_model(model, test_loader, target_cols)
    
    # 8. Visualisations
    plot_training(train_losses, val_losses)
    
    # 9. Sauvegarde
    save_model_info(model, scaler_X, feature_cols, target_cols, train_losses, val_losses)
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print("=" * 70)
    print("\n💡 Le modèle Transformer est plus puissant que le LSTM !")
    print("💡 Prochaine étape : Mettre à jour l'app pour utiliser ce modèle")

if __name__ == "__main__":
    main()