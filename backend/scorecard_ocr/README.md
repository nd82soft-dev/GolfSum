# GolfSum Scorecard OCR (Hybrid Pipeline v1)

This service parses photographed golf scorecards using a grid-first OCR pipeline.

## Tech
- Python 3.11
- FastAPI
- OpenCV
- Gemini 2.5 Flash (OCR/vision extraction)

## Endpoints
- `GET /health`
- `POST /scorecard/parse`

### POST /scorecard/parse
- Content-Type: `multipart/form-data`
- Form field: `image` (image/jpeg or image/png)

Response:
```json
{
  "confidence": 0.96,
  "holes": [{"hole":1,"par":4,"handicap_men":11,"handicap_women":11,"yardages_by_tee":{}}],
  "totals": {"out": 36, "in": 36, "total": 72},
  "flags": [],
  "metadata": {
    "tee_boxes": [],
    "rating_men_by_tee": {},
    "slope_men_by_tee": {},
    "rating_women_by_tee": {},
    "slope_women_by_tee": {}
  }
}
```

## Local Run
```bash
cd backend/scorecard_ocr
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notes
- OCR is performed via Gemini 2.5 Flash and then post-processed by the grid parser.
- Symbols are detected in a minimal heuristic pass.
