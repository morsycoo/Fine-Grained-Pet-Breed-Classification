# ============================================================
# Fine-Grained Pet Breed Classification
# FastAPI Production Inference
# ============================================================

from pathlib import Path
import io

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException


# ============================================================
# 1. Paths
# ============================================================

# app.py is located inside:
# Fine-Grained Pet Breed Classification/app/app.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "best_model.pth"


# ============================================================
# 2. Class Mapping
# ============================================================

class_names = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
]

NUM_CLASSES = len(class_names)


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. Model
# ============================================================

def build_model():
    """
    Build EfficientNet-B0 with the same classifier
    used during training.
    """

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=NUM_CLASSES,
            bias=True
        )
    )

    return model


# ============================================================
# 5. Load Final Model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )


efficientnet_b0 = build_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Support both plain state_dict and checkpoint dictionaries
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint


# Handle checkpoints saved using DataParallel
if any(
    key.startswith("module.")
    for key in state_dict.keys()
):
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }


efficientnet_b0.load_state_dict(
    state_dict
)

efficientnet_b0 = efficientnet_b0.to(device)
efficientnet_b0.eval()


# ============================================================
# 6. Production Preprocessing
# ============================================================

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. Prediction Function
# ============================================================

def predict_single(image: Image.Image) -> dict:
    """
    Run inference on a single PIL image.
    """

    # Ensure RGB
    image = image.convert("RGB")

    # Preprocessing
    image_tensor = inference_transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():

        outputs = efficientnet_b0(
            image_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = probabilities.max(
            dim=1
        )

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    return {
        "breed": class_names[predicted_class],
        "confidence": round(confidence, 4)
    }


# ============================================================
# 8. FastAPI Application
# ============================================================

app = FastAPI(
    title="Fine-Grained Pet Breed Classifier",
    description=(
        "EfficientNet-B0 inference API "
        "for 37 pet breeds."
    ),
    version="1.0.0"
)


# ============================================================
# 9. Health Endpoint
# ============================================================

@app.get("/health")
def health():
    """
    Check API and model availability.
    """

    return {
        "status": "ok",
        "model": "EfficientNet-B0",
        "device": str(device),
        "classes": NUM_CLASSES
    }


# ============================================================
# 10. Prediction Endpoint
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    """
    Predict the breed from an uploaded image.
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided."
        )

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
        "image/gif",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image type. "
                "Supported formats: JPEG, PNG, WEBP, "
                "BMP, TIFF, GIF."
            )
        )

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # --------------------------------------------------------
    # Open and validate image
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Force image loading to detect corrupted files
        image.load()

        image = image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        result = predict_single(image)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return result


# ============================================================
# 11. Startup Information
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 70)
    print("Fine-Grained Pet Breed Classification API")
    print("=" * 70)
    print(f"Model       : EfficientNet-B0")
    print(f"Checkpoint  : {MODEL_PATH}")
    print(f"Device      : {device}")
    print(f"Classes     : {NUM_CLASSES}")
    print("Model       : Ready")
    print("=" * 70)
