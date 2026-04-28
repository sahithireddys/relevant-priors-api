# Relevant Priors API

FastAPI solution for the relevant-priors-v1 challenge.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoint:

```text
POST /predict
```

Health check:

```text
GET /health
```

## Local contract test

```bash
pip install pytest httpx
pytest -q
```

## Public-calibrated statistics

The submitted zip already includes `app/learned_pair_stats.json`, generated from the public split.
To rebuild it after downloading the public JSON:

```bash
python train_public_stats.py relevant_priors_public.json
```

## Local public evaluation

After downloading the public eval JSON from the challenge page:

```bash
python eval_local.py public_eval.json
```

## Deploy

This app works on Render, Railway, Fly.io, or any server that can run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

For Render:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
