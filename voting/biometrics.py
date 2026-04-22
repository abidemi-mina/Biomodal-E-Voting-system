"""
Biometric Engine
================
Handles all facial recognition and liveness detection operations.

Components:
    - FaceNetEncoder: Generates 128-dim embeddings using FaceNet (facenet-pytorch)
    - LivenessDetector: MobileNetV2-based binary classifier (live vs spoof)
    - BiometricVerifier: Orchestrates registration and authentication

Dependencies:
    pip install facenet-pytorch torch torchvision opencv-python-headless Pillow numpy scipy
"""

import io
import os
import base64
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Try importing heavy ML dependencies; fall back to stub mode
# so that Django can still run/migrate without GPU/pytorch
# ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as tv_models
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
    import cv2
    BIOMETRIC_AVAILABLE = True
    logger.info("Biometric libraries loaded successfully.")
except ImportError as e:
    BIOMETRIC_AVAILABLE = False
    logger.warning(f"Biometric libraries not available: {e}. Running in STUB mode.")


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if (BIOMETRIC_AVAILABLE and torch.cuda.is_available()) else "cpu"
EMBEDDING_SIZE = 512          # InceptionResnetV1 output dimension
COSINE_THRESHOLD = 0.85       # Identity match threshold (lower = stricter)
LIVENESS_THRESHOLD = 0.70     # Live probability threshold


# ─────────────────────────────────────────────────────────────
# FaceNet Encoder
# ─────────────────────────────────────────────────────────────
class FaceNetEncoder:
    """
    Wraps facenet-pytorch InceptionResnetV1 pretrained on VGGFace2.
    Produces 512-dimensional L2-normalized face embeddings.
    """

    def __init__(self):
        if not BIOMETRIC_AVAILABLE:
            self._available = False
            return
        self._available = True
        # MTCNN for face detection and alignment
        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=DEVICE,
            keep_all=False,
        )
        # InceptionResnetV1 pretrained on VGGFace2
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        logger.info(f"FaceNet model loaded on {DEVICE}")

    def encode(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Takes raw image bytes, detects face, returns 512-dim embedding.
        Returns None if no face detected or library unavailable.
        """
        if not self._available:
            # Stub: return a random embedding for testing
            return np.random.rand(EMBEDDING_SIZE).astype(np.float32)

        try:
            img = _bytes_to_pil(image_data)
            if img is None:
                return None

            img_rgb = img.convert('RGB')

            # Detect and align face
            face_tensor = self.mtcnn(img_rgb)
            if face_tensor is None:
                logger.warning("No face detected in image.")
                return None

            # Generate embedding
            with torch.no_grad():
                face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
                embedding = self.model(face_tensor)
                embedding = embedding.cpu().numpy()[0]

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"FaceNet encoding error: {e}")
            return None

    def encode_multiple(self, images: List[bytes]) -> Optional[np.ndarray]:
        """
        Encodes multiple images and returns their mean embedding.
        Used during voter registration for a robust template.
        """
        embeddings = []
        for img_data in images:
            emb = self.encode(img_data)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None

        mean_embedding = np.mean(embeddings, axis=0)
        # Re-normalize after averaging
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        return mean_embedding.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Liveness Detector
# ─────────────────────────────────────────────────────────────
class LivenessDetector:
    """
    Binary liveness classifier (live=1, spoof=0) built on MobileNetV2.

    Architecture:
        MobileNetV2 (ImageNet pretrained) -> Dropout(0.5) -> Linear(1280, 256)
        -> ReLU -> Dropout(0.3) -> Linear(256, 2) -> Softmax

    In a production system this model should be fine-tuned on an anti-spoofing
    dataset such as CelebA-Spoof, LCC-FASD, or CASIA-FASD.
    The weights file path is set via LIVENESS_WEIGHTS_PATH.
    If no weights file is found the model runs in RANDOM STUB mode (for dev).
    """

    WEIGHTS_PATH = Path(__file__).parent.parent / 'media' / 'liveness_model.pth'

    def __init__(self):
        if not BIOMETRIC_AVAILABLE:
            self._available = False
            return
        self._available = True
        self._build_model()
        self._load_weights()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _build_model(self):
        base = tv_models.mobilenet_v2(weights=None)
        # Replace classifier
        base.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
        self.model = base.to(DEVICE)

    def _load_weights(self):
        if self.WEIGHTS_PATH.exists():
            try:
                state = torch.load(self.WEIGHTS_PATH, map_location=DEVICE)
                self.model.load_state_dict(state)
                self.model.eval()
                self._stub = False
                logger.info("Liveness model weights loaded.")
            except Exception as e:
                logger.warning(f"Could not load liveness weights: {e}. Using stub mode.")
                self._stub = True
        else:
            logger.warning("Liveness weights not found. Using stub mode (always returns live).")
            self._stub = True
            self.model.eval()

    def predict(self, image_data: bytes) -> Tuple[bool, float]:
        """
        Returns (is_live: bool, confidence: float 0-1).
        """
        if not self._available or self._stub:
            # Stub: assume live for development
            return True, 0.92

        try:
            img = _bytes_to_pil(image_data)
            if img is None:
                return False, 0.0

            img_rgb = img.convert('RGB')
            tensor = self.transform(img_rgb).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                live_prob = probs[0][1].item()  # class 1 = live

            is_live = live_prob >= LIVENESS_THRESHOLD
            return is_live, round(live_prob, 4)

        except Exception as e:
            logger.error(f"Liveness detection error: {e}")
            return False, 0.0


# ─────────────────────────────────────────────────────────────
# Biometric Verifier (Orchestrator)
# ─────────────────────────────────────────────────────────────
class BiometricVerifier:
    """
    High-level API used by Django views.
    Combines FaceNet encoding + liveness detection.
    """

    def __init__(self):
        self.encoder = FaceNetEncoder()
        self.liveness = LivenessDetector()

    # ── Registration ─────────────────────────────────────────
    def register(self, images: List[bytes]) -> dict:
        """
        Processes registration images.
        Returns {'success': bool, 'embedding': bytes | None, 'message': str}
        """
        if len(images) < 1:
            return {'success': False, 'embedding': None, 'message': 'No images provided.'}

        embedding = self.encoder.encode_multiple(images)
        if embedding is None:
            return {
                'success': False,
                'embedding': None,
                'message': 'Face not detected in provided images. Ensure good lighting and face is clearly visible.',
            }

        # Serialize embedding to bytes for DB storage
        embedding_bytes = pickle.dumps(embedding)
        return {
            'success': True,
            'embedding': embedding_bytes,
            'message': 'Biometric template created successfully.',
        }

    # ── Authentication ────────────────────────────────────────
    def authenticate(self, image_data: bytes, stored_embedding_bytes: bytes) -> dict:
        """
        Full authentication pipeline:
            1. Liveness detection
            2. Face encoding
            3. Cosine similarity against stored template

        Returns {
            'authenticated': bool,
            'liveness_pass': bool,
            'liveness_score': float,
            'similarity': float,
            'message': str,
        }
        """
        result = {
            'authenticated': False,
            'liveness_pass': False,
            'liveness_score': 0.0,
            'similarity': 0.0,
            'message': '',
        }

        # Step 1 – Liveness
        is_live, liveness_score = self.liveness.predict(image_data)
        result['liveness_score'] = liveness_score
        result['liveness_pass'] = is_live

        if not is_live:
            result['message'] = (
                f'Liveness check failed (score: {liveness_score:.2f}). '
                'Please ensure you are using a live camera, not a photo or screen.'
            )
            return result

        # Step 2 – Encode
        query_embedding = self.encoder.encode(image_data)
        if query_embedding is None:
            result['message'] = 'Face not detected. Ensure your face is clearly visible and well-lit.'
            return result

        # Step 3 – Compare
        try:
            stored_embedding = pickle.loads(stored_embedding_bytes)
        except Exception:
            result['message'] = 'Stored biometric template is corrupted.'
            return result

        similarity = float(cosine_similarity(query_embedding, stored_embedding))
        result['similarity'] = round(similarity, 4)

        if similarity >= COSINE_THRESHOLD:
            result['authenticated'] = True
            result['message'] = f'Identity verified (similarity: {similarity:.2f}).'
        else:
            result['message'] = (
                f'Face does not match registered voter (similarity: {similarity:.2f}, '
                f'threshold: {COSINE_THRESHOLD}). Please try again.'
            )

        return result


# ─────────────────────────────────────────────────────────────
# Metric utilities
# ─────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_metrics(true_labels: List[int], scores: List[float], threshold: float) -> dict:
    """
    Computes FAR, FRR, Accuracy, Precision, Recall, F1, EER.
    true_labels: 1=genuine, 0=impostor
    scores: cosine similarity scores
    """
    predictions = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)

    total_genuine = tp + fn
    total_impostor = tn + fp

    far = fp / total_impostor if total_impostor > 0 else 0
    frr = fn / total_genuine if total_genuine > 0 else 0
    accuracy = (tp + tn) / len(true_labels) if true_labels else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # EER approximation
    eer = (far + frr) / 2

    return {
        'accuracy': round(accuracy * 100, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'far': round(far * 100, 4),
        'frr': round(frr * 100, 4),
        'eer': round(eer * 100, 4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _bytes_to_pil(data: bytes):
    """Convert raw bytes (JPEG/PNG or base64-encoded) to PIL Image."""
    try:
        from PIL import Image
        # Handle base64-encoded images from frontend
        if isinstance(data, str):
            if ',' in data:
                data = data.split(',')[1]
            data = base64.b64decode(data)
        elif isinstance(data, bytes) and data[:4] in (b'data', b'iVBO'):
            try:
                data = base64.b64decode(data)
            except Exception:
                pass  # already raw bytes
        return Image.open(io.BytesIO(data))
    except Exception as e:
        logger.error(f"Image conversion error: {e}")
        return None


def base64_to_bytes(b64_string: str) -> bytes:
    """Convert a base64 data URL or plain base64 to bytes."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    return base64.b64decode(b64_string)


# Singleton instances (loaded once per process)
_verifier_instance = None

def get_verifier() -> BiometricVerifier:
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = BiometricVerifier()
    return _verifier_instance
