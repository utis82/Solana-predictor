from huggingface_hub import hf_hub_download
import os

print("📥 Téléchargement du modèle depuis Hugging Face...")

# Créer le dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Télécharger les fichiers
repo_id = "kookavicks/solana-transformer-model"

files = ['best_model.pth', 'scaler.pkl', 'metadata.json']

for file in files:
    print(f"⬇️  Téléchargement de {file}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        local_dir='models',
        local_dir_use_symlinks=False
    )
    print(f"✅ {file} téléchargé !")

print("🎉 Tous les fichiers du modèle sont prêts !")