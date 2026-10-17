# Multimodal Medical Document Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Système de **classification automatique et hiérarchique de documents médicaux hétérogènes**, combinant analyse textuelle (OCR + **CamemBERT**) et vision par ordinateur (**ResNet50**) au sein d'une architecture **multimodale par fusion tardive (Late Fusion)**, servie par une API **FastAPI** et une interface web interactive.

---

## 📌 Contexte Académique

Ce projet a été réalisé dans le cadre d'un **projet tutoré académique** à l'**École Nationale des Sciences Appliquées (ENSA) de Khouribga** :
- **Binôme d'élèves ingénieures :** Kadidiatou Dem & Hiba Chach
- **Encadrant :** Zakariyaa ISMAILI
- **Domaine :** Intelligence Artificielle appliquée à la santé, Traitement Automatique du Langage Naturel (NLP) et Vision par Ordinateur.

---

## 🎯 Problématique & Objectifs

Dans le secteur médical et hospitalier, les flux de documents sont extrêmement hétérogènes : comptes-rendus dactylographiés, certificats manuscrits, ordonnances, bilans biologiques et clichés d'imagerie médicale (scanners, radiographies, IRM). 

L'objectif de ce projet est de concevoir un système automatisé capable de :
1. **Identifier la nature du document** parmi 8 catégories médicales et administratives (**Niveau 1**).
2. **Déterminer la modalité anatomique spécifique** en cas de document d'imagerie (**Niveau 2**).
3. **Surmonter les limites unimodales** (OCR imparfait sur documents scannés ou absence de texte sur les radiographies) grâce à la fusion multimodale texte + vision.
4. **Déployer une solution opérationnelle** via une API REST rapide et une interface utilisateur ergonomique.

---

## 📊 Données Utilisées

Le système s'appuie sur deux sources de données complémentaires :

### 1. Corpus Expérimental Pseudo-Médical (adapté de RVL-CDIP)
En raison de la stricte confidentialité des données de santé réelles et de la réglementation RGPD, un corpus expérimental a été constitué à partir du benchmark public **RVL-CDIP** (*Ryerson Vision Lab Complex Document Information Processing*). Un mapping thématique a été établi pour projeter des typologies documentaires vers un référentiel médical :

| Catégorie Cible (Niveau 1) | Classe RVL-CDIP d'Origine | Description / Exemple |
|---|---|---|
| `lettre_liaison` | `letter` | Correspondances médicales et courriers d'adressage |
| `certificat_medical` | `form` | Formulaires officiels, attestations |
| `administratif` | `email`, `invoice` | Factures hospitalières, correspondances administratives |
| `consultation` | `handwritten` | Notes d'observations et comptes-rendus manuscrits |
| `biologie` | `scientific report` | Résultats d'analyses et bilans de laboratoire |
| `imagerie` | `scientific publication` | Articles et planches iconographiques médicales |
| `ordonnance` | `memo` | Prescriptions et consignes thérapeutiques |
| `autre` | `specification`, `questionnaire` | Questionnaires patients, documents divers |

### 2. Dataset d'Imagerie Médicale (Medical MNIST)
Pour le niveau 2 de classification hiérarchique, nous exploitons le dataset public **Medical MNIST** (environ 58 954 images réparties en 6 modalités/zones anatomiques) :
- `AbdomenCT` (Scanner abdomino-pelvien)
- `BreastMRI` (IRM mammaire)
- `CXR` (Radiographie pulmonaire / Chest X-Ray)
- `ChestCT` (Scanner thoracique)
- `Hand` (Radiographie de la main)
- `HeadCT` (Scanner crânien)

> ⚠️ **Précision éthique importante :** Le projet utilise un corpus expérimental dérivé de jeux de données publics. Il ne contient **aucune donnée médicale privée réelle** de patients.

---

## 🧠 Approches & Architecture Technique

```
                              ┌───────────────────────────────────┐
                              │     Document entrant (Image)      │
                              └─────────────────┬─────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
       ┌───────────────────────────────┐                 ┌───────────────────────────────┐
       │     Prétraitement OpenCV      │                 │  Prétraitement Albumentations  │
       │    & Extraction OCR Tesseract │                 │  (Resize 224x224, ImageNet)   │
       └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                       │                                                 │
          Longueur texte < 20 car. ?                                     │
           ├── [OUI] ──► Bypass direct Niveau 2                          │
           └── [NON]                                                     │
                       ▼                                                 ▼
       ┌───────────────────────────────┐                 ┌───────────────────────────────┐
       │        Branche Texte          │                 │        Branche Visuelle       │
       │    CamemBERT (Token [CLS])    │                 │      ResNet50 Backbone        │
       │         (1 x 768-d)           │                 │         (1 x 2048-d)          │
       └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │     Late Fusion (Concaténation) │
                               │           (1 x 2816-d)          │
                               └────────────────┬────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │   Classifieur MLP (512 -> 8)    │
                               └────────────────┬────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │    Prédiction Niveau 1 (L1)     │
                               └────────────────┬────────────────┘
                                                │
                       Classe == "imagerie" OU Confiance < 0.60 ?
                                       ├── [NON] ──► Réponse Finale L1
                                       └── [OUI]
                                                ▼
                               ┌─────────────────────────────────┐
                               │       ResNet50 (Niveau 2)       │
                               │     Spécialisé Medical MNIST    │
                               │           (6 classes)           │
                               └─────────────────────────────────┘
```

### 1. Prétraitement & OCR
- Pipeline de débruitage avec `cv2.fastNlMeansDenoising` et binarisation adaptative d'Otsu.
- Extraction textuelle multilingue (`fra+eng`) via **Tesseract OCR**.
- Si le texte extrait comporte moins de 20 caractères (typique d'une radiographie pure sans en-tête dactylographié), le système effectue un **bypass direct** vers le classifieur d'imagerie Niveau 2.

### 2. Branche Visuelle (ResNet50)
- Architecture ResNet50 pré-entraînée sur ImageNet, fine-tunée sur les images de documents (couche finale projetée en espace 2048-d via `nn.Identity()`).
- Augmentation de données avec `Albumentations` (redimensionnement 224x224, normalisation standard).

### 3. Branche Textuelle (CamemBERT)
- Modèle de langue pré-entraîné `camembert-base` adapté au vocabulaire français.
- Tokenisation avec padding/truncation à 256 tokens.
- Extraction de l'embedding contextuel global issu du token `[CLS]` (768 dimensions).

### 4. Fusion Multimodale & MLP
- Concaténation des deux vecteurs de caractéristiques : $768 + 2048 = 2816$ dimensions.
- Perceptron multicouche (MLP) : `Linear(2816, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 8)`.
- Entraînement conjoint sur GPU avec l'optimiseur **AdamW** (learning rate $2 \times 10^{-5}$) et **CrossEntropyLoss**.

### 5. Cascade Hiérarchique (Niveau 2)
- Si le document est prédit comme `imagerie` ou si l'indice de confiance du Niveau 1 est inférieur à $0.60$, le modèle spécialisé **ResNet50 Medical MNIST** est invoqué.
- Si la confiance du Niveau 2 dépasse $0.80$, la prédiction est automatiquement raffinée avec la sous-classe radiologique (`AbdomenCT`, `BreastMRI`, `CXR`, etc.).

---

## 📈 Résultats Expérimentaux

Les évaluations menées au fil des phases de développement démontrent l'apport décisif de la multimodalité par rapport aux modèles univariés :

| Modèle / Approche | Modalité | Données | Métrique Principale | Observation |
|---|---|---|---|---|
| **Baseline OCR + ML** (Phase 1) | Texte seul | RVL-CDIP (8 classes) | **F1-score : 74.0%** (Acc. 73.9%) | Dégradé par la qualité variable de l'OCR sur les scans bruités |
| **ResNet50 Feature Extractor** | Vision seule | RVL-CDIP (8 classes) | **F1-score : ~59%** (Acc. 60.0%) | Couches convolutionnelles gelées insuffisantes pour la mise en page |
| **ResNet50 Fine-Tuning** | Vision seule | RVL-CDIP (8 classes) | **F1-score : 80.0%** (Acc. 79.8%) | Bonne capture de la structure spatiale globale des documents |
| **Fusion CamemBERT + ResNet50** | **Multimodal** | RVL-CDIP (8 classes) | **F1-score : 85.24%** (Acc. 85.25%) | **Meilleure performance globale (+5.2 pts vs vision, +11.2 pts vs texte)** |
| **ResNet50 Medical MNIST (Niveau 2)** | Vision seule | Medical MNIST (6 classes) | **F1-score : 100%** (Acc. 100%) | Séparation quasi-parfaite des modalités radiologiques |

---

## 📂 Structure du Répertoire

```
multimodal-medical-document-classification/
├── app/
│   ├── __init__.py            # Module Python de l'API
│   └── main.py                # Serveur FastAPI (chargement modèles, pipeline de prédiction L1/L2)
├── frontend/
│   └── index.html             # Interface Web moderne (Drag & Drop, jauges, scores détaillés)
├── notebook/
│   ├── Analyseexp_Phase1(ocr+ml).ipynb   # Notebook 1 : EDA, OCR et modélisation baseline ML
│   └── notebookfa80748ae7 (14).ipynb     # Notebook 2 : Deep Learning, Fine-Tuning et Fusion Multimodale
├── weights/
│   └── README.md              # Instructions sur les fichiers de checkpoints (.pt)
├── .gitignore                 # Règles d'exclusion des poids, environnements et fichiers temporaires
├── requirements.txt           # Dépendances Python du projet
├── ARCHITECTURE.md            # Spécifications détaillées de l'architecture technique
└── README.md                  # Ce fichier
```

---

## ⚙️ Installation & Configuration

### Prérequis
- **Python 3.10 à 3.12**
- **Tesseract OCR** :
  - Sur Windows : installer Tesseract (ex. `Tesseract-OCR/tesseract.exe` ou chemin système)
  - Sur Linux/Ubuntu : `sudo apt-get install tesseract-ocr tesseract-ocr-fra`
  - Sur macOS : `brew install tesseract tesseract-lang`

### 1. Cloner le dépôt
```bash
git clone https://github.com/votre-identifiant/multimodal-medical-document-classification.git
cd multimodal-medical-document-classification
```

### 2. Créer un environnement virtuel
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Fichiers de poids (`weights/`)
> ⚠️ **Note sur les poids d'entraînement :** En raison de leur taille (~850 Mo au total), les checkpoints `.pt` ne sont pas versionnés sur GitHub. Consultez le fichier [`weights/README.md`](weights/README.md) pour les détails sur les fichiers requis :
- `weights/best_multimodal.pt` (Modèle multimodal CamemBERT + MLP)
- `weights/resnet_fineT_model_rvl.pt` (ResNet50 RVL Feature Extractor)
- `weights/resnet_fineT_model_med (1).pt` (ResNet50 Medical MNIST niveau 2)

---

## 🚀 Utilisation & Lancement

### 1. Démarrer le serveur backend FastAPI
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Le serveur s'initialise et charge les modèles en mémoire (détection automatique GPU CUDA si disponible, sinon CPU).
- Documentation interactive Swagger : `http://127.0.0.1:8000/docs`
- Statut de l'API : `http://127.0.0.1:8000/health`
- Nomenclature des classes : `http://127.0.0.1:8000/version`

### 2. Lancer l'interface Frontend
Ouvrez directement le fichier `frontend/index.html` dans votre navigateur web, ou servez-le via un serveur statique :
```bash
# Depuis la racine du projet
python -m http.server 3000 --directory frontend
```
Puis accédez à `http://127.0.0.1:3000` :
- Glissez-déposez un document (PNG, JPEG, TIFF, BMP).
- Visualisez la classe prédite, la jauge de confiance, la sous-classe d'imagerie si applicable, le temps d'inférence en millisecondes et la distribution complète des probabilités.

---

## 🛠️ Technologies Utilisées

- **Deep Learning & NLP :** [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/), [Hugging Face Transformers](https://huggingface.co/) (CamemBERT), [Albumentations](https://albumentations.ai/)
- **Traitement d'Image & OCR :** [OpenCV](https://opencv.org/), [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), [PyTesseract](https://pypi.org/project/pytesseract/), [Pillow](https://python-pillow.org/)
- **Backend & Déploiement :** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic v2](https://docs.pydantic.dev/)
- **Frontend :** HTML5, CSS3 moderne (design sombre, CSS variables, glassmorphism), Vanilla JavaScript

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus d'informations.
