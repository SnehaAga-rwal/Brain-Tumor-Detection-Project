#!/usr/bin/env python3
"""
Diagnostic script for brain tumor ensemble inference.
Run from project root: python diagnose_model.py [image1.jpg image2.png ...]

Shows per-model softmax outputs, ensemble blend, preprocessing sanity checks,
and optional side-by-side visualization of resized RGB input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Brain tumor model diagnostics")
    parser.add_argument(
        "images",
        nargs="*",
        help="Paths to MRI images (PNG/JPEG). If omitted, uses test_data.pkl samples.",
    )
    parser.add_argument(
        "--save-viz",
        metavar="DIR",
        help="If set, save preprocessing visualization PNGs under this directory",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args()

    os.environ["BRAIN_TUMOR_DEBUG"] = "1"

    import matplotlib.pyplot as plt
    import numpy as np

    from app.models.ensemble_model import EnsemblePredictor

    predictor = EnsemblePredictor()
    status = predictor.get_model_status()
    if args.json:
        print(json.dumps({"model_status": status}, indent=2))
        if not status["resnet50"] and not status["mobilenetv2"] and not status["custom_cnn"]:
            sys.exit(2)
    else:
        print("=== Model load status ===")
        print(json.dumps(status, indent=2))

    image_paths = list(args.images)
    if not image_paths:
        pkl = os.path.join("data", "processed", "test_data.pkl")
        if os.path.isfile(pkl):
            import pickle

            with open(pkl, "rb") as f:
                X, _ycat, y = pickle.load(f)
            for label in (0, 1, 2, 3):
                idx = next((i for i in range(len(y)) if int(y[i]) == label), None)
                if idx is None:
                    continue
                import cv2

                img = (np.clip(X[idx], 0, 1) * 255).astype(np.uint8)
                out = f"_diag_sample_class{label}.png"
                cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                image_paths.append(out)
            print("No images passed; generated samples from test_data.pkl:", image_paths)
        else:
            print("No images and no data/processed/test_data.pkl — pass image paths.", file=sys.stderr)
            sys.exit(1)

    for path in image_paths:
        rgb = predictor.load_image_rgb_uint8(path)
        if not args.json:
            print(f"\n=== File: {path} ===")
            print(f"RGB tensor shape: {rgb.shape}, dtype={rgb.dtype}, min={rgb.min()}, max={rgb.max()}")

        r = predictor.predict(path, debug=True)

        if args.save_viz:
            os.makedirs(args.save_viz, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(rgb)
            ax[0].set_title("Resized RGB (224)")
            ax[0].axis("off")
            mn = predictor.rgb_to_model_batch(rgb, "custom_cnn")[0]
            # Visualize first channel of preprocessed tensor (not natural RGB)
            ax[1].imshow(mn[:, :, 0], cmap="viridis")
            ax[1].set_title("MobileNet preprocess (ch0)")
            ax[1].axis("off")
            viz_path = os.path.join(args.save_viz, f"{base}_preprocess.png")
            fig.savefig(viz_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            if not args.json:
                print(f"Saved visualization: {viz_path}")

        if args.json:
            print(json.dumps({"file": path, "result": r}, indent=2))
        else:
            print("Prediction:", r.get("prediction"), "confidence:", r.get("confidence"))
            print("Class probabilities (%):", r.get("class_probabilities"))
            print("Models used:", r.get("models_used"))
            if r.get("error"):
                print("ERROR:", r.get("error"))
            for name, det in (r.get("model_details") or {}).items():
                print(f"  [{name}] {det.get('prediction')} {det.get('confidence'):.2f}%")
                print(f"         softmax sum={sum((det.get('raw_probabilities') or [0])):.6f}")

    # Cleanup temp diag samples
    for p in image_paths:
        if p.startswith("_diag_sample_class") and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()
