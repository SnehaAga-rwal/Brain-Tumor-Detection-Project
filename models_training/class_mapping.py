import os

# Canonical order used across preprocessing, training, and inference
CANONICAL_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Accept common Kaggle/variant folder names for each canonical class
CLASS_ALIASES = {
    "glioma": ["glioma", "glioma_tumor", "glioma tumour", "glioma-tumor"],
    "meningioma": ["meningioma", "meningioma_tumor", "meningioma tumour", "meningioma-tumor"],
    "notumor": ["notumor", "no_tumor", "notumor_tumor", "no tumor", "normal"],
    "pituitary": ["pituitary", "pituitary_tumor", "pituitary tumour", "pituitary-tumor"],
}


def resolve_class_dirs(split_path: str) -> dict:
    """
    Resolve dataset directories into canonical class names.
    Returns: {canonical_class: absolute_dir_path}
    """
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Dataset split path not found: {split_path}")

    existing_dirs = {
        d.lower(): os.path.join(split_path, d)
        for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    }

    resolved = {}
    missing = []
    for canonical in CANONICAL_CLASSES:
        found = None
        for alias in CLASS_ALIASES[canonical]:
            if alias.lower() in existing_dirs:
                found = existing_dirs[alias.lower()]
                break
        if found:
            resolved[canonical] = found
        else:
            missing.append(canonical)

    if missing:
        raise ValueError(
            f"Missing class folders for {missing} under {split_path}. "
            f"Found folders: {sorted(existing_dirs.keys())}"
        )

    return resolved

