#!/bin/bash
echo "Téléchargement des fichiers Git LFS..."
git lfs fetch --all
git lfs pull
echo "Fichiers LFS téléchargés!"