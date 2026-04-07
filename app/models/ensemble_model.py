# ============================================
# FILE: app/models/ensemble_model.py
# ============================================
import os
import json
import logging
import threading
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

try:
    from tensorflow.keras import config as keras_config

    if hasattr(keras_config, "enable_unsafe_deserialization"):
        keras_config.enable_unsafe_deserialization()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Filenames produced by training scripts (see models_training/train_*.py)
MODEL_CANDIDATES = {
    "resnet50": [
        "resnet50_best.h5",
        "resnet50_best.keras",
        "resnet50_95.keras",
        "resnet50_simple.keras",
        "resnet50_final.keras",
    ],
    "mobilenetv2": [
        "mobilenetv2_best.h5",
        "mobilenetv2_best.keras",
        "mobilenetv2_final.keras",
        "brain_tumor_best.keras",
        "mobilenetv2_finetuned.keras",
    ],
    "custom_cnn": [
        "custom_cnn_best.h5",
        "custom_cnn_final.h5",
    ],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _discover_model_dirs() -> list[Path]:
    """Ordered search paths for weight files."""
    roots = [
        Path(__file__).resolve().parent / "saved",
        Path(os.getcwd()) / "app" / "models" / "saved",
        _project_root() / "app" / "models" / "saved",
        _project_root() / "saved_models",
        Path(os.getcwd()) / "saved_models",
        Path(os.getcwd()) / "models_training" / "saved",
    ]
    seen = set()
    out = []
    for p in roots:
        rp = p.resolve()
        if rp not in seen and rp.is_dir():
            seen.add(rp)
            out.append(rp)
    return out


def _find_model_path(model_key: str) -> str | None:
    for base in _discover_model_dirs():
        for name in MODEL_CANDIDATES.get(model_key, []):
            path = base / name
            if path.is_file():
                return str(path)
    return None


def _load_keras_model(path: str):
    """load_model with best compatibility across TF/Keras versions."""
    try:
        return models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return models.load_model(path, compile=False)


class EnsemblePredictor:
    """Loads trained models and performs weighted ensemble prediction."""

    def __init__(self, models_dir=None):
        self.logger = logging.getLogger(__name__)
        self.debug = os.environ.get("BRAIN_TUMOR_DEBUG", "").lower() in ("1", "true", "yes")
        self.img_size = 224
        self.class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        self.display_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

        self.models_dir = models_dir or str(_discover_model_dirs()[0])
        self.config = self._load_config()

        self.resnet_model = None
        self.mobilenet_model = None
        self.custom_model = None

        self._load_models()
        self._log_load_status()

    def _load_config(self):
        config_path = os.path.join(self.models_dir, "ensemble_config.json")
        default_config = {
            # ResNet / Custom CNN often agree; MobileNet can disagree — lower its blend weight
            "weights": [0.35, 0.25, 0.40],
            "class_names": self.class_names,
            "accuracy": 0.965,
            "model_accuracies": {
                "resnet50": 0.948,
                "mobilenetv2": 0.952,
                "custom_cnn": 0.945,
            },
            # Final label: majority if >= min_models_agree else blended argmax
            "use_agreement_confidence": True,
            "min_models_agree": 2,
            "confidence_metric": "max_support",
            # Optional display floors (0–100). Set to null in JSON to disable.
            # When >= min_models_agree models share the same top-1 class, headline confidence is at least this value.
            "min_confidence_when_models_agree": None,
            # Only apply majority floor if max P(final class) across models is at least this (avoids 95% when two heads are ~55%).
            "majority_floor_min_max_support": 0.85,
            # When any model assigns >= this fraction to the final predicted class, headline is at least strong_model_pct.
            "min_confidence_when_max_support_at_least": None,
            "min_confidence_strong_model_pct": None,
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except OSError:
                return default_config
        return default_config

    def _load_models(self):
        order = [("resnet50", "resnet_model"), ("mobilenetv2", "mobilenet_model"), ("custom_cnn", "custom_model")]
        for key, attr in order:
            path = _find_model_path(key)
            if not path:
                self.logger.warning("No weight file found for %s (searched: %s)", key, MODEL_CANDIDATES.get(key))
                continue
            try:
                model = _load_keras_model(path)
                setattr(self, attr, model)
                self.logger.info("Loaded %s from %s", key, path)
            except Exception as e:
                self.logger.warning("Could not load %s from %s: %s", key, path, e)

    def _log_load_status(self):
        st = self.get_model_status()
        self.logger.info(
            "Model load status: resnet=%s mobilenet=%s custom=%s",
            st["resnet50"],
            st["mobilenetv2"],
            st["custom_cnn"],
        )

    def load_image_rgb_uint8(self, image_path: str) -> np.ndarray:
        """Load image as RGB uint8 HxWx3; supports common MRI formats and optional DICOM."""
        path_lower = image_path.lower()
        if path_lower.endswith((".dcm", ".dicom")):
            try:
                import pydicom
                from pydicom.pixel_data_handlers.util import apply_voi_lut
            except ImportError:
                self.logger.error(
                    "DICOM file requires pydicom. Install with: pip install pydicom"
                )
                raise
            try:
                ds = pydicom.dcmread(image_path)
                arr = ds.pixel_array.astype(np.float32)
                if hasattr(ds, "WindowWidth") and hasattr(ds, "WindowCenter"):
                    arr = apply_voi_lut(arr, ds)
                else:
                    arr = arr - arr.min()
                    if arr.max() > 0:
                        arr = arr / arr.max()
                    arr = (arr * 255).astype(np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.ndim == 3 and arr.shape[-1] > 3:
                    arr = arr[:, :, 0]
                    arr = np.stack([arr, arr, arr], axis=-1)
                return cv2.resize(arr, (self.img_size, self.img_size))
            except Exception as e:
                self.logger.error("DICOM read failed (%s), trying OpenCV/PIL: %s", image_path, e)

        img = cv2.imread(image_path)
        if img is None:
            from PIL import Image

            pil = Image.open(image_path).convert("RGB")
            img = np.array(pil)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return cv2.resize(img, (self.img_size, self.img_size))

    def rgb_to_model_batch(self, rgb_uint8: np.ndarray, model_key: str) -> np.ndarray:
        """
        Training used ImageNet-style preprocessing for all three heads:
        - custom_cnn (train_custom_cnn.py): mobilenet_v2.preprocess_input(X * 255)
        - mobilenetv2 / resnet: same family preprocessing as in train_mobilenetv2 / train_resnet50
        """
        x = rgb_uint8.astype(np.float32)
        if x.max() <= 1.0:
            x = x * 255.0
        x = np.expand_dims(x, axis=0)
        if model_key in ("custom_cnn", "mobilenetv2"):
            return mobilenet_preprocess_input(x.copy())
        if model_key == "resnet50":
            return resnet_preprocess_input(x.copy())
        raise ValueError(f"Unknown model_key: {model_key}")

    def preprocess_image(self, image_path):
        """Backward-compatible name: returns MobileNet-style batch (used by custom_cnn / default)."""
        rgb = self.load_image_rgb_uint8(image_path)
        return self.rgb_to_model_batch(rgb, "custom_cnn")

    def _prob_dict(self, probs: np.ndarray) -> dict:
        return {n: float(p * 100.0) for n, p in zip(self.class_names, probs)}

    def _validate_softmax(self, probs: np.ndarray, label: str) -> float:
        s = float(np.sum(probs))
        if self.debug or abs(s - 1.0) > 0.02:
            self.logger.info("[%s] softmax_sum=%.6f raw=%s", label, s, np.array2string(probs, precision=4))
        return s

    def predict(self, image_path, debug: bool | None = None):
        """Run ensemble prediction; returns per-class probabilities and diagnostics."""
        dbg = self.debug if debug is None else debug
        # Fresh config each scan (weights / confidence_metric) without restarting the process singleton
        self.config = self._load_config()
        if not any([self.resnet_model, self.mobilenet_model, self.custom_model]):
            self.logger.error("No Keras models loaded — cannot run inference.")
            return self._error_result("No neural network weights could be loaded. Check app/models/saved and logs.")

        try:
            rgb = self.load_image_rgb_uint8(image_path)
        except Exception as e:
            self.logger.exception("Failed to read image: %s", image_path)
            return self._error_result(f"Could not read image: {e}")

        weights = self.config.get("weights", [0.35, 0.25, 0.40])
        models_to_check = [
            ("resnet50", self.resnet_model, weights[0]),
            ("mobilenetv2", self.mobilenet_model, weights[1]),
            ("custom_cnn", self.custom_model, weights[2]),
        ]

        model_preds = {}
        weighted_slices = []
        active_weights = []

        for name, model, w in models_to_check:
            if not model:
                continue
            batch = self.rgb_to_model_batch(rgb, name)
            probs = model.predict(batch, verbose=0)[0]
            self._validate_softmax(probs, name)
            pred_idx = int(np.argmax(probs))
            entry = {
                "prediction": self.class_names[pred_idx],
                "confidence": float(probs[pred_idx] * 100.0),
                "probabilities": self._prob_dict(probs),
                "raw_probabilities": probs.astype(float).tolist(),
            }
            model_preds[name] = entry
            weighted_slices.append(probs * w)
            active_weights.append(w)

        if not model_preds:
            return self._error_result("All model.predict calls failed.")

        wsum = float(np.sum(active_weights))
        ensemble_probs = np.sum(weighted_slices, axis=0) / wsum
        self._validate_softmax(ensemble_probs, "ensemble")
        ens_idx = int(np.argmax(ensemble_probs))
        blended_confidence_pct = float(ensemble_probs[ens_idx] * 100.0)

        use_agreement = self.config.get("use_agreement_confidence", True)
        min_agree = int(self.config.get("min_models_agree", 2))
        conf_metric = str(self.config.get("confidence_metric", "max_support")).lower().strip()

        vote_list = [model_preds[n]["prediction"] for n in model_preds]
        vote_counts = Counter(vote_list)
        majority_class, majority_count = vote_counts.most_common(1)[0]

        if majority_count >= min_agree and use_agreement:
            prediction = majority_class
        else:
            prediction = self.class_names[ens_idx]

        pred_idx_final = self.class_names.index(prediction)
        support_per_model = []
        for n in model_preds:
            raw = np.asarray(model_preds[n]["raw_probabilities"], dtype=np.float64)
            support_per_model.append(float(raw[pred_idx_final]))

        max_support_pct = float(np.max(support_per_model) * 100.0) if support_per_model else blended_confidence_pct

        if conf_metric == "blend":
            confidence = blended_confidence_pct
            confidence_source = "weighted_softmax_blend"
        elif conf_metric == "mean_support":
            confidence = float(np.mean(support_per_model) * 100.0) if support_per_model else blended_confidence_pct
            confidence_source = "mean_prob_for_final_class_across_models"
        else:
            confidence = max_support_pct
            confidence_source = "max_prob_for_final_class_across_models"

        # Headline confidence must never be below the strongest backbone's belief in the chosen class.
        # (Blended softmax often sits ~55–65% when one model disagrees — see pituitary vs notumor case.)
        if max_support_pct > confidence:
            confidence = max_support_pct
            confidence_source = f"{confidence_source}_raised_to_max_support"

        confidence_before_floor = confidence

        # Optional floors so the UI can show >=95% when the ensemble strongly supports the chosen label
        # (not a medical calibration — see README).
        floor_agree = self.config.get("min_confidence_when_models_agree")
        maj_evidence = float(self.config.get("majority_floor_min_max_support", 0.85))
        if floor_agree is not None and use_agreement and majority_count >= min_agree:
            try:
                fa = float(floor_agree)
                strong_enough = support_per_model and max(support_per_model) >= maj_evidence
                if 0 < fa <= 100 and strong_enough:
                    confidence = min(100.0, max(confidence, fa))
                    confidence_source = f"{confidence_source}_floor_majority_{fa:g}"
            except (TypeError, ValueError):
                pass

        thr_strong = self.config.get("min_confidence_when_max_support_at_least")
        pct_strong = self.config.get("min_confidence_strong_model_pct")
        if thr_strong is not None and pct_strong is not None and support_per_model:
            try:
                thr = float(thr_strong)
                cap = float(pct_strong)
                if 0 < thr <= 1 and 0 < cap <= 100 and max(support_per_model) >= thr:
                    confidence = min(100.0, max(confidence, cap))
                    confidence_source = f"{confidence_source}_floor_strong_{cap:g}"
            except (TypeError, ValueError):
                pass

        out = {
            "prediction": prediction,
            "confidence": confidence,
            "ensemble_confidence": confidence,
            "class_probabilities": {c: float(ensemble_probs[i] * 100.0) for i, c in enumerate(self.class_names)},
            "all_probabilities": {
                d: float(ensemble_probs[i] * 100.0) for i, d in enumerate(self.display_names)
            },
            "model_predictions": {k: {"prediction": v["prediction"], "confidence": v["confidence"]} for k, v in model_preds.items()},
            "model_details": model_preds,
            "models_used": list(model_preds.keys()),
            "preprocessing": "ImageNet preprocess_input per backbone (MobileNet for custom+mobilenet, ResNet50 for resnet)",
            "debug": {
                "image_shape": list(rgb.shape),
                "ensemble_weight_sum": wsum,
                "active_weights": active_weights,
                "votes": vote_list,
                "majority_class": majority_class,
                "majority_count": majority_count,
                "confidence_metric": conf_metric,
                "confidence_source": confidence_source,
                "support_for_final_class_per_model": dict(zip(model_preds.keys(), support_per_model)),
                "blended_top1_class": self.class_names[ens_idx],
                "blended_confidence_pct": blended_confidence_pct,
                "max_support_pct": max_support_pct,
                "confidence_before_floor": confidence_before_floor,
            },
        }
        if dbg:
            self.logger.info("Ensemble prediction for %s: %s", image_path, json.dumps(out["class_probabilities"]))
        return out

    def _error_result(self, message: str):
        """Explicit failure state (no fake 'no tumor' at high confidence)."""
        z = {c: 0.0 for c in self.class_names}
        return {
            "prediction": "notumor",
            "confidence": 0.0,
            "ensemble_confidence": 0.0,
            "error": message,
            "class_probabilities": z,
            "all_probabilities": {"Glioma": 0.0, "Meningioma": 0.0, "No Tumor": 0.0, "Pituitary": 0.0},
            "model_predictions": {},
            "model_details": {},
            "models_used": [],
            "preprocessing": None,
            "debug": {"error": message},
        }

    def _fallback_prediction(self, image_path):
        """Deprecated demo path; only used if explicitly needed. Prefer _error_result."""
        self.logger.warning("Using filename fallback for %s — not clinically valid.", image_path)
        filename = os.path.basename(image_path).lower()
        prediction = "notumor"
        confidence = 15.0
        if "pi" in filename:
            prediction = "pituitary"
        elif "me" in filename:
            prediction = "meningioma"
        elif "gl" in filename:
            prediction = "glioma"
        fake = {n: 25.0 for n in self.class_names}
        if prediction in fake:
            fake[prediction] = 70.0
        return {
            "prediction": prediction,
            "confidence": confidence,
            "ensemble_confidence": confidence,
            "warning": "filename_heuristic_only",
            "class_probabilities": fake,
            "all_probabilities": {
                "Glioma": fake["glioma"],
                "Meningioma": fake["meningioma"],
                "No Tumor": fake["notumor"],
                "Pituitary": fake["pituitary"],
            },
            "model_predictions": {},
            "model_details": {},
            "models_used": [],
        }

    def get_model_status(self):
        return {
            "resnet50": self.resnet_model is not None,
            "mobilenetv2": self.mobilenet_model is not None,
            "custom_cnn": self.custom_model is not None,
            "ensemble_weights": self.config.get("weights", [0.3, 0.4, 0.3]),
            "accuracy": self.config.get("accuracy", 0.965),
            "models_dir": self.models_dir,
        }


_cached_predictor: EnsemblePredictor | None = None
_predictor_lock = threading.Lock()


def get_ensemble_predictor() -> EnsemblePredictor:
    """Single shared predictor so routes and /health do not reload weights repeatedly."""
    global _cached_predictor
    if _cached_predictor is not None:
        return _cached_predictor
    with _predictor_lock:
        if _cached_predictor is None:
            _cached_predictor = EnsemblePredictor()
    return _cached_predictor
