# Scorecard OCR Regression Suite

Place regression images and expected outputs here. Each case should have:

- image file (e.g. `.jpg`)
- expected JSON output (e.g. `.json`)
- notes in `cases.json`

This suite is used by `backend/scorecard_ocr/tests/test_regression.py`.

## Directory layout

```
regression/scorecards/
  cases.json
  case-ambiguous-score.jpg
  case-ambiguous-score.expected.json
  case-missing-rating.jpg
  case-missing-rating.expected.json
  ...
```

Add new cases by updating `cases.json`.
