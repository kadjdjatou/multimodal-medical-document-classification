# Model Checkpoints (`weights/`)

> **Note:** Les fichiers de poids pré-entraînés ne sont pas inclus dans ce dépôt Git en raison de leur taille volumineuse (~850 Mo au total).

Pour exécuter l'API FastAPI et le pipeline d'inférence en local, placez les fichiers suivants dans ce répertoire :

| Fichier attendu | Description | Taille approx. | Rôle dans l'application |
|---|---|---|---|
| `best_multimodal.pt` | Poids entraînés du modèle multimodal (CamemBERT + MLP) | ~448 Mo | Modèle principal de fusion texte + image (Niveau 1) |
| `resnet_fineT_model_rvl.pt` | Checkpoint ResNet50 fine-tuné sur RVL-CDIP (8 classes) | ~94 Mo | Extracteur de features visuelles (2048-d) pour le Niveau 1 |
| `resnet_fineT_model_med (1).pt` | Checkpoint ResNet50 fine-tuné sur Medical MNIST (6 classes) | ~94 Mo | Classifieur hiérarchique de modalité d'imagerie (Niveau 2) |

### Checkpoints d'expérimentation additionnels (Notebooks) :
- `resnet_feature_rvl.pt` (~94 Mo) : ResNet50 extraction de caractéristiques (couche fc gelée)
- `resnet_feature_med.pt` (~94 Mo) : ResNet50 extraction de caractéristiques Medical MNIST
