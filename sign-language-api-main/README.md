# Sign Language API

Flask API: webcam frame -> MediaPipe hand landmarks -> TFLite Transformer -> predicted sign.

## Run
```bash
pip install -r requirements.txt
python app.py                      # dev
# production (keep --workers 1: frame buffers live in memory)
gunicorn app:app --workers 1 --threads 4 --timeout 120
```

## Endpoints
- `POST /predict_frame` `{ "image": "<base64 jpeg/png>", "session_id": "<unique per user/tab>" }`
  - `status`: `buffering` | `success` | `low_confidence`
  - `prediction`, `confidence` once 30 frames are collected
- `POST /reset` `{ "session_id": "..." }` clears that session's buffer
- `GET /health`

## Config (env vars)
`PORT`, `MODEL_PATH`, `LABELS_PATH` (default `labels.json`), `CONFIDENCE_THRESHOLD` (0.88),
`MAX_HANDS` (1; use 2 if signs need both hands), `ALLOWED_ORIGINS` (default `*`),
`SESSION_TTL_SECONDS` (60), `FRAMES` (30).

## Labels
Add a `labels.json` (JSON array of sign names, in the same order as the model's classes), e.g.
`["Hello", "Thanks", "Yes"]`. Without it you get placeholders like `Sign_0`.
