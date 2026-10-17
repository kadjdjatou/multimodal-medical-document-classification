# Architecture du Projet & Pipeline d'Inférence

Ce document décrit l'organisation du projet ainsi que l'architecture technique du pipeline de classification multimodale hiérarchique.

---

## 1. Arborescence du Dépôt

```
multimodal-medical-document-classification/
├── app/
│   ├── __init__.py            # Initialisation du module Python
│   └── main.py                # Application FastAPI complète (chargement modèles, OCR, inférence L1/L2, endpoints)
├── frontend/
│   └── index.html             # Interface Web moderne (Drag & Drop, visualisation confiance, modalité d'imagerie)
├── notebook/
│   ├── Analyseexp_Phase1(ocr+ml).ipynb   # Phase 1 : Exploration, prétraitement OCR, baseline ML
│   └── notebookfa80748ae7 (14).ipynb     # Phases 2 à 4 : Transfer Learning ResNet50 & Fusion multimodale CamemBERT
├── weights/
│   └── README.md              # Documentation des fichiers de poids (.pt) requis
├── .gitignore                 # Exclusion des fichiers volumineux, .venv, Tesseract et caches
├── requirements.txt           # Dépendances Python nécessaires
├── ARCHITECTURE.md            # Ce document d'architecture technique
└── README.md                  # Documentation principale du projet
```

---

## 2. Pipeline de Classification Hiérarchique

```mermaid
flowchart TD
    A[Document entrant : Image/Scan/Radio] --> B[Prétraitement OpenCV & Extraction OCR Tesseract]
    
    B --> C{Longueur texte < 20 caractères ?}
    
    %% Cas radiographie pure / peu de texte
    C -- "Oui (ex: Cliché pur)" --> D["Bypass Niveau 1\nRoutage direct vers Niveau 2"]
    
    %% Cas document multimodal standard
    C -- "Non (Texte suffisant)" --> E["Branche Texte : Tokenisation CamemBERT (Embedding [CLS] 768-d)"]
    A --> F["Branche Visuelle : Prétraitement Albumentations + ResNet50 (Embedding 2048-d)"]
    
    E --> G["Concaténation (Late Fusion : 2816-d)"]
    F --> G
    
    G --> H["Classifieur MLP (2816 → 512 → 8 classes)"]
    H --> I[Prédiction Niveau 1]
    
    %% Décision de niveau 2
    I --> J{"Classe = 'imagerie'\nOU Confiance < 0.60 ?"}
    J -- Oui --> D
    J -- Non --> K[Réponse Finale Niveau 1]
    
    %% Niveau 2
    D --> L["ResNet50 Fine-Tuné (Medical MNIST : 6 classes)"]
    L --> M{Confiance L2 >= 0.80 ?}
    M -- Oui --> N["Attribution sous-classe imagerie\n(AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT)"]
    M -- Non --> O[Conservation de la classe L1 initiale]
    
    N --> P[Réponse Finale Enrichie]
    O --> P
```

---

## 3. Endpoints de l'API FastAPI

| Méthode | Route | Description | Entrée | Sortie |
|---|---|---|---|---|
| `GET` | `/health` | Statut du service et device actif (`cpu` ou `cuda`) | Aucune | `{"status": "ok", "device": "...", "model_version": "v1.0.0"}` |
| `GET` | `/version` | Version du modèle et listes des classes L1 / L2 | Aucune | `VersionResponse` avec nomenclature des 8 classes L1 et 6 classes L2 |
| `POST` | `/predict` | Inférence complète sur document téléversé | Fichier image (`image/png`, `image/jpeg`, etc.) | `PredictionResponse` (classe prédite, score, scores détaillés, sous-classe d'imagerie, temps ms) |
