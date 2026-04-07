# Brain Tumor Detection System

Flask web application for MRI scan upload, **multi-class AI diagnosis** (four outcomes), doctor review, and reporting. The AI stack is a **weighted ensemble** of ResNet50, MobileNetV2, and a custom MobileNetV2-based CNN.

### Confidence score (headline %)

The **ensemble class scores** row still shows the **weighted softmax blend** per class (so you can see disagreement). The **headline confidence** is computed in `app/models/saved/ensemble_config.json` as follows:

1. **Predicted class** — **majority vote** among the three top-1 labels when at least `min_models_agree` models match; otherwise the **blended** argmax class.

2. **Base metric (`confidence_metric`)**  
   - **`max_support`** (default): `max` over models of softmax for the **final** class (avoids ~65% blend when one backbone is ~99% sure).  
   - **`mean_support`**: mean of those three values.  
   - **`blend`**: blended softmax for the top class only.

3. **Optional display floors** (UI-oriented; not a medical calibration):  
   - **`min_confidence_when_models_agree`** (e.g. `95`) — if majority agrees, raise headline to at least this **only when** `max` support for that class ≥ **`majority_floor_min_max_support`** (default `0.85`), so two weak ~55% agreements do not become 95%.  
   - **`min_confidence_when_max_support_at_least`** + **`min_confidence_strong_model_pct`** (e.g. `0.92` and `95`) — if any model assigns ≥ 92% to the final class, headline is at least 95% (helps when there is no majority but one head is very confident).

Set any floor key to `null` in JSON to disable it. **`debug.confidence_before_floor`** in stored diagnosis JSON shows the value before floors. **Re-run inference** (new upload) after changing config; old DB rows keep previous numbers.

The UI, PDFs, and emails use **`diagnosis|headline_confidence`** (`app/utils/confidence_display.py`): the shown value is at least the **maximum** of the stored DB value and each model’s softmax for the **predicted** class (using `debug.max_support_pct` when present, else `raw_probabilities` in `model_details`). So older scans that only stored the **blended** ~57% still display **~100%** when ResNet assigns ~100% to pituitary.

## Class labels (four types)

The models predict one of:

| Internal ID   | Meaning        |
|---------------|----------------|
| `glioma`      | Glioma tumor   |
| `meningioma`  | Meningioma     |
| `notumor`     | No tumor       |
| `pituitary`   | Pituitary tumor|

Training folder layout matches `data/raw/Training/{glioma,meningioma,notumor,pituitary}/`.

## Deploying on Azure (~$100 student credit)

See **[DEPLOY_AZURE.md](DEPLOY_AZURE.md)** for App Service, credits, RAM notes, and optional Docker. Use **gunicorn** in production (included in `requirements.txt`).

## Quick start (Windows)

1. **Python 3.10+** recommended (project tested with TensorFlow 2.20).

2. Create and activate a virtual environment, then install dependencies:

   ```text
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. From the **project root** (`brain_tumor_detection_system`), run:

   ```text
   python run.py
   ```

4. Open a browser: **http://localhost:5000**

5. **Default accounts** (created on first run):

   | Role    | Email                      | Password    |
   |---------|----------------------------|-------------|
   | Admin   | admin@braintumor.com       | Admin@123   |
   | Patient | patient@braintumor.com     | Patient@123 |
   | Doctor  | doctor@braintumor.com     | Doctor@123  |

If the console shows encoding issues on older Windows terminals, `run.py` configures UTF-8 on stdout when possible; startup messages use plain ASCII.

## Model weights (required for real predictions)

Place trained files under **`app/models/saved/`**. The loader recognizes common names, including:

- ResNet: `resnet50_95.keras`, `resnet50_simple.keras`, `resnet50_best.h5`, …
- MobileNetV2: `mobilenetv2_final.keras`, `brain_tumor_best.keras`, `mobilenetv2_best.keras`, …
- Custom CNN: `custom_cnn_best.h5`, `custom_cnn_final.h5`

Optional extra search paths: `saved_models/`, `models_training/saved/`.

### Verify models loaded

- **GET** http://localhost:5000/health — JSON with `status: "ok"` when all three weights are loaded.

## Features overview

- **Patient**: Login, dashboard, **upload MRI** (PNG, JPG, JPEG, DICOM), view scan status and AI result with **per-class probabilities**, history, treatment PDF download, profile, notifications.
- **Doctor**: Dashboard, **review AI cases**, submit diagnosis and treatment plan, patient list, reports, analytics.
- **Admin**: User management (as implemented in `admin_routes`).

AI analysis runs **in a background thread** after upload; refresh the scan page until status is **completed**.

## MRI upload and preprocessing

- Images are resized to **224×224 RGB**.
- Preprocessing matches training: **ImageNet `preprocess_input`** for MobileNet heads and **ResNet50 `preprocess_input`** for the ResNet head.
- **DICOM** (`.dcm`): install `pydicom` (`requirements.txt`). Conversion uses windowing when present.

## Diagnostics CLI

From project root:

```text
python diagnose_model.py
```

Uses `data/processed/test_data.pkl` samples if no paths are given. Optional:

```text
python diagnose_model.py path\to\image1.png --save-viz visualization_outputs
python diagnose_model.py --json
```

Set **`BRAIN_TUMOR_DEBUG=1`** for verbose softmax logging in the app.

## Training and data pipeline

1. Put the public brain-tumor MRI dataset (or your own with the same class folders) under `data/raw/Training` and `data/raw/Testing`.
2. Run `models_training/data_preprocessor.py` to build `data/processed/*.pkl`.
3. Train with `models_training/train_custom_cnn.py`, `train_mobilenetv2.py`, `train_resnet50.py`, or your ensemble script; save weights into `app/models/saved/`.

## Limitations (important for demos)

- This is a **research / educational** system, not a medical device. **Never** use output alone for clinical decisions.
- Accuracy depends on **image quality**, **dataset match**, and whether **all three** weights load. Use `/health` and `diagnose_model.py` to confirm.
- The ensemble can disagree across backbones; the UI shows individual model outputs where available.

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| Always “No tumor” or 0% confidence | `/health` — are all three models `true`? Missing files → degraded mode. |
| Slow first prediction | TensorFlow loads large weights once; subsequent scans reuse the same predictor. |
| Unicode error on `python run.py` | Use latest `run.py` (UTF-8 reconfigure + ASCII banner). |
| DICOM fails | `pip install pydicom` |

## License / attribution

Use your own license. If you use a public dataset, cite it according to its terms.
