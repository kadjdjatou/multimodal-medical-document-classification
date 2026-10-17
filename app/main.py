"""
Phase 5 — API FastAPI — Classification de documents médicaux
Compatible avec le notebook d'entraînement (notebookfa80748ae7).

Lancement :
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

import time
import io
import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pytesseract
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from transformers import CamembertTokenizer, CamembertModel

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Tesseract (Windows) — décommentez si besoin ───────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_VERSION            = "v1.0.0"
DEVICE                   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_TEXT_LENGTH          = 20
IMAGING_CONF_THRESHOLD   = 0.60
IMAGING_ACCEPT_THRESHOLD = 0.80

# ── Classes niveau 1 — ordre alphabétique = sorted() du notebook ──────────────
# notebook : labels = sorted(df["project_class"].unique())
#            label2id = {label: idx for idx, label in enumerate(labels)}
# → {'administratif':0,'autre':1,'biologie':2,'certificat_medical':3,
#    'consultation':4,'imagerie':5,'lettre_liaison':6,'ordonnance':7}
CLASS_NAMES_L1 = [
    "administratif",       # 0
    "autre",               # 1
    "biologie",            # 2
    "certificat_medical",  # 3
    "consultation",        # 4
    "imagerie",            # 5
    "lettre_liaison",      # 6
    "ordonnance",          # 7
]
# ── Classes niveau 2 — ordre alphabétique = sorted() du notebook ──────────────
# notebook : classes_med = sorted([d.name for d in MED_PATH.iterdir() if d.is_dir()])
# → ['AbdomenCT','BreastMRI','CXR','ChestCT','Hand','HeadCT']
CLASS_NAMES_L2 = [
    "AbdomenCT",  # 0
    "BreastMRI",  # 1
    "CXR",        # 2
    "ChestCT",    # 3
    "Hand",       # 4
    "HeadCT",     # 5
]
# ── Chemins des poids ─────────────────────────────────────────────────────────
MULTIMODAL_PATH  = "weights/best_multimodal.pt"
IMAGE_MODEL_PATH = "weights/resnet_fineT_model_rvl.pt"
IMAGING_PATH     = "weights/resnet_fineT_model_med (1).pt"

# ── Modèles globaux ───────────────────────────────────────────────────────────
text_model    = None  # CamembertModel
image_model   = None  # ResNet50 feature extractor
classifier    = None  # MLP 2816 → 512 → 8
imaging_model = None  # ResNet50 Medical MNIST
tokenizer     = None  # CamembertTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    predicted_class: str
    confidence: float
    all_scores: dict
    model_version: str
    processing_time_ms: float
    imaging_subclass: Optional[str] = None
    imaging_subclass_confidence: Optional[float] = None
    imaging_all_scores: Optional[dict] = None
    imaging_forced: Optional[bool] = None


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    device: str
    model_version: str


class VersionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_version: str
    level1_classes: list
    level2_classes: list
    device: str


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — identique au val_transform du notebook (Albumentations)
# val_transform = A.Compose([A.Resize(224,224),
#                 A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
#                 ToTensorV2()])
# Les deux ResNets (RVL + Medical MNIST) partagent ce même transform.
# ─────────────────────────────────────────────────────────────────────────────
_val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Transform identique pour les deux ResNets (RVL et Medical MNIST)."""
    img_np = np.array(pil_image.convert("RGB"))
    return _val_transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────
def extract_text(pil_image: Image.Image) -> str:
    try:
        img_cv    = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray      = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised  = cv2.fastNlMeansDenoising(gray, h=10)
        _, binary = cv2.threshold(denoised, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(
            Image.fromarray(binary), lang="fra+eng"
        ).strip()
    except Exception as e:
        logger.warning("OCR échoué : %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation — identique à MultimodalDataset.__getitem__ du notebook
# encoding = tokenizer(text, padding="max_length", truncation=True,
#                      max_length=256, return_tensors="pt")
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text: str):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Inférence niveau 2
# ─────────────────────────────────────────────────────────────────────────────
def run_level2(pil_image: Image.Image):
    with torch.no_grad():
        logits = imaging_model(preprocess_image(pil_image))
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    logger.info("L2 scores : %s",
                {c: f"{p:.3f}" for c, p in zip(CLASS_NAMES_L2, probs)})
    return (
        CLASS_NAMES_L2[idx],
        float(probs[idx]),
        {c: float(p) for c, p in zip(CLASS_NAMES_L2, probs)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic checkpoint — affiche les infos clés au démarrage
# ─────────────────────────────────────────────────────────────────────────────
def _diagnose_checkpoint(path: str, expected_fc_out: int):
    """
    Charge le checkpoint et vérifie que la couche fc correspond
    au nombre de classes attendu. Logue un avertissement si ce n'est pas le cas.
    """
    ckpt = torch.load(path, map_location="cpu")
    clean = {k.replace("model.", ""): v for k, v in ckpt.items()}

    fc_weight = clean.get("fc.weight")
    if fc_weight is not None:
        actual_out = fc_weight.shape[0]
        status = "✅" if actual_out == expected_fc_out else "  MAUVAIS FICHIER"
        logger.info(
            "  %s fc.weight shape = [%d, 2048]  (attendu : %d sorties)",
            status, actual_out, expected_fc_out
        )
        if actual_out != expected_fc_out:
            logger.warning(
                "  → Le checkpoint '%s' a %d sorties fc "
                "mais le modèle attend %d. "
                "Vérifiez que vous utilisez le bon fichier .pt !",
                path, actual_out, expected_fc_out
            )
    else:
        logger.warning("   Clé 'fc.weight' introuvable dans %s", path)

    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ─────────────────────────────────────────────────────────────────────────────
def _load_resnet_feature_extractor(path: str, num_classes: int) -> nn.Module:
    """
    Charge ResNetFineTuning comme feature extractor (fc → Identity).
    Séquence exacte du notebook Phase 4 :
      1. net.fc = Linear(2048, num_classes)
      2. load_state_dict(clean_sd, strict=False)
      3. net.fc = Identity()
    """
    logger.info("── Diagnostic %s (attendu : %d classes fc) ──", path, num_classes)
    ckpt = _diagnose_checkpoint(path, num_classes)
    clean_sd = {k.replace("model.", ""): v for k, v in ckpt.items()}

    net = models.resnet50(weights=None)
    net.fc = nn.Linear(2048, num_classes)
    missing, unexpected = net.load_state_dict(clean_sd, strict=False)
    logger.info("  missing=%d  unexpected=%d", len(missing), len(unexpected))
    if missing:
        logger.warning("  Clés manquantes : %s", missing[:5])

    net.fc = nn.Identity()  # → sortie (batch, 2048)
    return net.to(DEVICE).eval()


def _load_resnet_classifier(path: str, num_classes: int) -> nn.Module:
    """
    Charge ResNetFineTuning comme classifieur final (fc garde ses poids).
    Utilisé pour imaging_model (Medical MNIST, 6 classes).
    """
    logger.info("── Diagnostic %s (attendu : %d classes fc) ──", path, num_classes)
    ckpt = _diagnose_checkpoint(path, num_classes)
    clean_sd = {k.replace("model.", ""): v for k, v in ckpt.items()}

    net = models.resnet50(weights=None)
    net.fc = nn.Linear(2048, num_classes)
    missing, unexpected = net.load_state_dict(clean_sd, strict=False)
    logger.info("  missing=%d  unexpected=%d", len(missing), len(unexpected))
    if missing:
        logger.warning("  Clés manquantes : %s", missing[:5])

    return net.to(DEVICE).eval()


def load_models():
    global text_model, image_model, classifier, imaging_model, tokenizer

    logger.info("=" * 60)
    logger.info("CHARGEMENT DES MODÈLES  (device=%s)", DEVICE)
    logger.info("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info("Chargement tokenizer CamemBERT…")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # ── 1. Multimodal checkpoint (text_model + classifier) ────────────────
    logger.info("── Chargement %s ──", MULTIMODAL_PATH)
    ckpt = torch.load(MULTIMODAL_PATH, map_location=DEVICE)
    logger.info("  Clés disponibles : %s", list(ckpt.keys()))

    text_model = CamembertModel.from_pretrained("camembert-base")
    text_model.load_state_dict(ckpt["text_model"])
    text_model = text_model.to(DEVICE).eval()
    logger.info("  text_model chargé  (val_f1=%.4f, epoch=%d)",
                ckpt.get("val_f1", 0.0), ckpt.get("epoch", -1))

    # Architecture MLP exacte du notebook :
    # nn.Sequential(nn.Linear(2816,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,8))
    classifier = nn.Sequential(
        nn.Linear(2816, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 8),
    ).to(DEVICE)
    classifier.load_state_dict(ckpt["classifier"])
    classifier.eval()
    logger.info("   classifier MLP chargé")

    # ── 2. ResNet RVL — feature extractor (8 classes → Identity) ─────────
    image_model = _load_resnet_feature_extractor(IMAGE_MODEL_PATH, num_classes=8)
    logger.info(" image_model (ResNet50 RVL feature extractor) prêt")

    # ── 3. ResNet Medical MNIST — classifieur final (6 classes) ──────────
    #   Si le diagnostic affiche un avertissement ici, le fichier .pt
    #     utilisé n'est pas le bon (ex: fichier RVL chargé à la place).
    imaging_model = _load_resnet_classifier(IMAGING_PATH, num_classes=6)
    logger.info(" imaging_model (ResNet50 Medical MNIST) prêt")

    logger.info("=" * 60)
    logger.info("Tous les modèles sont chargés. API prête.")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Classificateur Documents Médicaux", version=MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    load_models()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", device=str(DEVICE),
                          model_version=MODEL_VERSION)


@app.get("/version", response_model=VersionResponse)
async def version():
    return VersionResponse(
        model_version=MODEL_VERSION,
        level1_classes=CLASS_NAMES_L1,
        level2_classes=CLASS_NAMES_L2,
        device=str(DEVICE),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    t_start = time.perf_counter()

    # Validation format
    allowed = {"image/png", "image/jpeg", "image/tiff", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(415,
            detail=f"Format non supporté : {file.content_type}")

    try:
        contents  = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, detail=f"Impossible de lire l'image : {e}")

    # OCR
    text = extract_text(pil_image)
    logger.info("OCR → %d caractères", len(text))

    # ══ CAS 1 : peu de texte → niveau 2 direct ════════════════════════════
    if len(text) < MIN_TEXT_LENGTH:
        logger.info("Bypass niveau 1 → imagerie directe")
        subclass, sub_conf, sub_scores = run_level2(pil_image)
        ms = (time.perf_counter() - t_start) * 1000
        return PredictionResponse(
            predicted_class="imagerie",
            confidence=round(sub_conf, 4),
            all_scores={
                c: (round(sub_conf, 4) if c == "imagerie" else 0.0)
                for c in CLASS_NAMES_L1
            },
            model_version=MODEL_VERSION,
            processing_time_ms=round(ms, 1),
            imaging_subclass=subclass,
            imaging_subclass_confidence=round(sub_conf, 4),
            imaging_all_scores={k: round(v, 4) for k, v in sub_scores.items()},
            imaging_forced=True,
        )

    # ══ CAS 2 : niveau 1 multimodal ═══════════════════════════════════════
    with torch.no_grad():
        # Branche texte
        input_ids, attn_mask = tokenize(text)
        text_out = text_model(input_ids=input_ids, attention_mask=attn_mask)
        text_emb = text_out.last_hidden_state[:, 0, :]   # (1, 768)  token [CLS]

        # Branche image
        img_emb = image_model(preprocess_image(pil_image))  # (1, 2048)

        # Fusion et classification
        combined  = torch.cat([text_emb, img_emb], dim=1)   # (1, 2816)
        logits_l1 = classifier(combined)
        probs_l1  = torch.softmax(logits_l1, dim=1).squeeze().cpu().numpy()

    logger.info("Logits L1 : %s", logits_l1.detach().cpu().numpy().round(2))
    logger.info("Probs  L1 : %s", probs_l1.round(3))

    idx_l1          = int(np.argmax(probs_l1))
    predicted_class = CLASS_NAMES_L1[idx_l1]
    confidence_l1   = float(probs_l1[idx_l1])
    all_scores_l1   = {c: float(p) for c, p in zip(CLASS_NAMES_L1, probs_l1)}

    # Niveau 2 conditionnel
    imaging_subclass = imaging_subclass_confidence = imaging_all_scores = None
    imaging_forced   = False

    if predicted_class == "imagerie" or confidence_l1 < IMAGING_CONF_THRESHOLD:
        subclass, sub_conf, sub_scores = run_level2(pil_image)
        imaging_forced = predicted_class != "imagerie"

        if sub_conf >= IMAGING_ACCEPT_THRESHOLD:
            imaging_subclass            = subclass
            imaging_subclass_confidence = sub_conf
            imaging_all_scores          = sub_scores
            if imaging_forced:
                predicted_class = "imagerie"
        else:
            imaging_forced = False

    ms = (time.perf_counter() - t_start) * 1000
    logger.info("predict | class=%s conf=%.3f sub=%s time=%.0fms",
                predicted_class, confidence_l1, imaging_subclass, ms)

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=round(confidence_l1, 4),
        all_scores={k: round(v, 4) for k, v in all_scores_l1.items()},
        model_version=MODEL_VERSION,
        processing_time_ms=round(ms, 1),
        imaging_subclass=imaging_subclass,
        imaging_subclass_confidence=(
            round(imaging_subclass_confidence, 4)
            if imaging_subclass_confidence is not None else None
        ),
        imaging_all_scores=(
            {k: round(v, 4) for k, v in imaging_all_scores.items()}
            if imaging_all_scores else None
        ),
        imaging_forced=imaging_forced if imaging_forced else None,
    )