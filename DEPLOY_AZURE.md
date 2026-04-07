# Deploying on Microsoft Azure with ~$100 school credits

This project is a **Flask + TensorFlow** app with **large model files** (ResNet / MobileNet / custom CNN). The **Free** Azure App Service tier is usually **too small** (memory ~1 GB). Plan for a **paid Linux App Service** (or small VM) and **stop** resources when you are not demoing to stretch credits.

## 1. Get Microsoft / Azure student credits

- Sign in at [Azure for Students](https://azure.microsoft.com/free/students/) (or your school’s Microsoft Azure Education offer).
- Verify your **school email** / status. You typically get **~$100 credit** (region and program details vary; no credit card required for many student offers).
- In the [Azure Portal](https://portal.azure.com), open **Cost Management + Billing → Credits** to watch remaining balance.

Also check **[GitHub Student Pack](https://education.github.com/pack)** — it sometimes includes extra Azure or cloud credits.

## 2. What will use your credits?

Rough order of cost drivers:

| Item | Notes |
|------|--------|
| **App Service** (Linux) | B2 or similar so RAM fits TensorFlow + 3 models (~**2 GB+** safer). |
| **Storage** | Model `.keras` / `.h5` files in `app/models/saved/` — keep in deployment package or Azure Files. |
| **Database** | **SQLite** is cheapest (file on disk) but **resets** if the app is moved/reimaged; **Azure Database for PostgreSQL** costs more but is durable. |
| **Outbound bandwidth** | Usually small for a class demo. |

**Stretch $100:** delete or **stop** the App Service when not in use; use **short demos**; avoid always-on high SKUs.

## 3. Recommended path: Azure App Service (Linux) + GitHub

### Prerequisites

- Push the project to **GitHub** (do **not** commit `venv/` or secrets).
- Ensure `app/models/saved/` contains your weights (or document download step). Repo size may be large — consider **Git LFS** or uploading models to **Azure Blob** and a startup script (advanced).

### Portal steps (high level)

1. **Create resource group** (e.g. `rg-brain-tumor-demo`).
2. **Create App Service**  
   - **Publish**: Code or Docker (this repo includes an optional `Dockerfile`).  
   - **Runtime**: Python **3.11** or **3.12** (match what you use locally; TensorFlow wheels must exist for that version).  
   - **Region**: nearest to you.  
   - **Pricing plan**: **Linux** — pick **B2** (or **P1v3** if B2 still OOMs during model load). Avoid F1 for this app.
3. **Configuration → Application settings** (environment variables), e.g.:

   | Name | Example |
   |------|--------|
   | `WEBSITES_PORT` | `8000` (if using gunicorn on 8000) |
   | `SECRET_KEY` | long random string |
   | `DATABASE_URL` | `sqlite:////home/site/wwwroot/instance/brain_tumor.db` (see SQLite note below) |
   | `FLASK_CONFIG` | `production` (add a `ProductionConfig` in `app/config.py` if you use this) |
   | `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` |

4. **General settings → Startup Command** (Linux):

   ```bash
   gunicorn --bind 0.0.0.0:8000 --timeout 300 --workers 1 run:app
   ```

   Long **timeout** helps first request while TensorFlow loads models. **Workers 1** saves RAM.

5. **Deployment Center** → connect **GitHub** → select repo/branch → save. Azure will build and deploy.

6. After deploy, open `https://<your-app>.azurewebsites.net` and test `/health`.

### SQLite on App Service

- Default SQLite path on Linux is often under `/home/site/wwwroot`.  
- Ensure the `instance/` folder exists and is writable (`mkdir` in startup or Oryx post-build).  
- Treat SQLite as **non-durable** across major redeploys unless you attach **persistent storage**.

### Production hardening (before real use)

- Set `DEBUG=False`, strong `SECRET_KEY`, HTTPS-only cookies in `ProductionConfig`.  
- Use **Azure Key Vault** or App Settings (not Git) for secrets.  
- Do **not** expose admin defaults in production.

## 4. Optional: Docker

From the repo root (with Docker installed):

```bash
docker build -t brain-tumor-app .
docker run -p 8000:8000 -e PORT=8000 brain-tumor-app
```

Deploy the same image to **Azure Container Apps** or **App Service for Containers** if you prefer containers (often similar or slightly higher cost than App Service).

## 5. If credits run low

- **Stop** the App Service in the portal (you stop billing for compute on many plans; confirm current Azure pricing).  
- Delete unused resource groups when the course is over.  
- For a **static demo only**, record a video instead of hosting 24/7.

## 6. This repo

- **`gunicorn`** is in `requirements.txt` for production serving.  
- **`Dockerfile`** (optional) builds a Linux image suitable for container hosting.  
- Entry point: **`run:app`** in `run.py`.

---

**Disclaimer:** Azure pricing and student programs change. Check [azure.microsoft.com/pricing](https://azure.microsoft.com/pricing/) and your school’s IT page. This app is for **education / research**, not a regulated medical device.
