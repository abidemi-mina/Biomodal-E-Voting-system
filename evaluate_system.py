"""
evaluate_system.py
==================
Evaluate the full biometric pipeline on a test dataset.
Computes standard biometric metrics: Accuracy, Precision, Recall,
F1-Score, FAR, FRR, EER.

Usage:
    python evaluate_system.py --test_dir /path/to/test_images --output results/metrics.json

Test directory structure:
    test_dir/
        genuine/        ← Pairs: registered_face vs authentic_attempt
            voter_001/
                registered.jpg
                attempt_01.jpg
                attempt_02.jpg
        impostor/       ← Pairs: registered_face vs wrong person attempt
            attack_001/
                registered.jpg
                impostor_attempt.jpg

Or use the --csv mode for pre-built test CSVs.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'evoting.settings')

import django
django.setup()

from voting.biometrics import get_verifier, cosine_similarity, compute_metrics, base64_to_bytes

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_image_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


def evaluate_on_directory(test_dir: str) -> Dict:
    """
    Evaluate genuine pairs and impostor pairs.
    Returns metrics dict.
    """
    verifier = get_verifier()
    test_dir = Path(test_dir)

    true_labels = []
    scores = []
    liveness_results = []

    # ── Genuine pairs ──────────────────────────────────────────
    genuine_dir = test_dir / 'genuine'
    if genuine_dir.exists():
        for voter_dir in sorted(genuine_dir.iterdir()):
            if not voter_dir.is_dir():
                continue
            registered = voter_dir / 'registered.jpg'
            if not registered.exists():
                continue
            attempts = [f for f in voter_dir.iterdir()
                        if f.name != 'registered.jpg' and f.suffix in ('.jpg', '.jpeg', '.png')]

            reg_bytes = load_image_bytes(registered)
            reg_result = verifier.register([reg_bytes])
            if not reg_result['success']:
                logger.warning(f"Could not encode registered face: {registered}")
                continue

            stored_emb = reg_result['embedding']
            for attempt in attempts:
                att_bytes = load_image_bytes(attempt)
                auth = verifier.authenticate(att_bytes, stored_emb)
                true_labels.append(1)  # genuine
                scores.append(auth['similarity'])
                liveness_results.append(auth['liveness_pass'])

    logger.info(f"Genuine pairs loaded: {sum(l==1 for l in true_labels)}")

    # ── Impostor pairs ─────────────────────────────────────────
    impostor_dir = test_dir / 'impostor'
    if impostor_dir.exists():
        for attack_dir in sorted(impostor_dir.iterdir()):
            if not attack_dir.is_dir():
                continue
            registered = attack_dir / 'registered.jpg'
            impostor_attempts = [f for f in attack_dir.iterdir()
                                 if f.name != 'registered.jpg' and f.suffix in ('.jpg', '.jpeg', '.png')]
            if not registered.exists():
                continue

            reg_bytes = load_image_bytes(registered)
            reg_result = verifier.register([reg_bytes])
            if not reg_result['success']:
                continue

            stored_emb = reg_result['embedding']
            for attempt in impostor_attempts:
                att_bytes = load_image_bytes(attempt)
                auth = verifier.authenticate(att_bytes, stored_emb)
                true_labels.append(0)  # impostor
                scores.append(auth['similarity'])
                liveness_results.append(auth['liveness_pass'])

    logger.info(f"Impostor pairs loaded: {sum(l==0 for l in true_labels)}")

    if not true_labels:
        logger.error("No test pairs found.")
        return {}

    # ── Compute metrics at multiple thresholds ─────────────────
    from django.conf import settings
    threshold = getattr(settings, 'FACENET_THRESHOLD', 0.85)

    metrics = compute_metrics(true_labels, scores, threshold)
    metrics['threshold'] = threshold
    metrics['total_genuine'] = sum(l == 1 for l in true_labels)
    metrics['total_impostor'] = sum(l == 0 for l in true_labels)
    metrics['total_samples'] = len(true_labels)
    metrics['liveness_pass_rate'] = round(sum(liveness_results) / len(liveness_results) * 100, 2)

    # EER threshold search
    thresholds = np.arange(0.5, 1.0, 0.01)
    best_eer = 1.0
    best_eer_threshold = threshold
    for t in thresholds:
        m = compute_metrics(true_labels, scores, float(t))
        eer_at_t = abs(m['far'] - m['frr'])
        if eer_at_t < abs(best_eer):
            best_eer = (m['far'] + m['frr']) / 2
            best_eer_threshold = float(t)
    metrics['eer'] = round(best_eer, 4)
    metrics['eer_threshold'] = round(best_eer_threshold, 2)

    return metrics


def print_report(metrics: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "═" * 60)
    print("  BIOMETRIC SYSTEM EVALUATION REPORT")
    print("  NigeriaVotes — FaceNet + Liveness Detection")
    print("═" * 60)
    print(f"  Total samples  : {metrics.get('total_samples', 0)}")
    print(f"  Genuine pairs  : {metrics.get('total_genuine', 0)}")
    print(f"  Impostor pairs : {metrics.get('total_impostor', 0)}")
    print(f"  Threshold used : {metrics.get('threshold', 'N/A')}")
    print("─" * 60)
    print(f"  Accuracy       : {metrics.get('accuracy', 0):.2f}%")
    print(f"  Precision      : {metrics.get('precision', 0):.2f}%")
    print(f"  Recall         : {metrics.get('recall', 0):.2f}%")
    print(f"  F1-Score       : {metrics.get('f1_score', 0):.2f}%")
    print("─" * 60)
    print(f"  FAR (False Acceptance Rate) : {metrics.get('far', 0):.4f}%")
    print(f"  FRR (False Rejection Rate)  : {metrics.get('frr', 0):.4f}%")
    print(f"  EER (Equal Error Rate)      : {metrics.get('eer', 0):.4f}%  (at threshold={metrics.get('eer_threshold', 'N/A')})")
    print("─" * 60)
    print(f"  Liveness Pass Rate          : {metrics.get('liveness_pass_rate', 0):.2f}%")
    print("─" * 60)
    print(f"  TP={metrics.get('tp',0)}  TN={metrics.get('tn',0)}  FP={metrics.get('fp',0)}  FN={metrics.get('fn',0)}")
    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate E-Voting Biometric System')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test dataset directory')
    parser.add_argument('--output', type=str, default='results/metrics.json',
                        help='Path to save JSON metrics report')
    args = parser.parse_args()

    logger.info(f"Evaluating on: {args.test_dir}")
    metrics = evaluate_on_directory(args.test_dir)

    if metrics:
        print_report(metrics)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {output_path}")
    else:
        logger.error("Evaluation failed — no metrics computed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
