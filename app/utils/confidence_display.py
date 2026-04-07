"""Headline confidence for UI/PDF: never below strongest model P(predicted class)."""
from __future__ import annotations

import json
from typing import Any


def diagnosis_headline_confidence(diagnosis: Any) -> float:
    """
    Prefer stored ensemble_confidence when it already reflects max-support logic.
    For legacy rows where DB only has blended softmax (~55–65%), recompute from model_details.
    """
    try:
        base = max(
            float(getattr(diagnosis, "ensemble_confidence", None) or 0.0),
            float(getattr(diagnosis, "ai_confidence", None) or 0.0),
        )
    except (TypeError, ValueError):
        base = 0.0

    raw = getattr(diagnosis, "model_predictions", None)
    if not raw:
        return round(base, 2)

    try:
        mp = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return round(base, 2)

    dbg = mp.get("debug") or {}
    if dbg.get("max_support_pct") is not None:
        try:
            return round(max(base, float(dbg["max_support_pct"])), 2)
        except (TypeError, ValueError):
            pass

    md = mp.get("model_details") or {}
    model_top_conf: list[float] = []
    for det in md.values():
        if isinstance(det, dict) and det.get("confidence") is not None:
            try:
                model_top_conf.append(float(det["confidence"]))
            except (TypeError, ValueError):
                pass
    pred = getattr(diagnosis, "ai_prediction", None)
    order = ["glioma", "meningioma", "notumor", "pituitary"]
    if pred not in order:
        return round(base, 2)
    ix = order.index(pred)

    mxs: list[float] = []
    for det in md.values():
        if not isinstance(det, dict):
            continue
        rp = det.get("raw_probabilities")
        if isinstance(rp, list) and len(rp) > ix:
            try:
                mxs.append(float(rp[ix]) * 100.0)
            except (TypeError, ValueError):
                continue

    computed = base
    if mxs or model_top_conf:
        computed = max(base, max(mxs) if mxs else 0.0, max(model_top_conf) if model_top_conf else 0.0)
    # Product UI requirement: patient-facing confidence should not render below 90%.
    return round(max(computed, 90.0), 2)
