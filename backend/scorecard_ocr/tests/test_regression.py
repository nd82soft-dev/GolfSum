import json
from pathlib import Path
import pytest

from app.pipeline import parse_scorecard_image

BASE_DIR = Path(__file__).resolve().parents[1]
REGRESSION_DIR = BASE_DIR / "regression" / "scorecards"
CASES_FILE = REGRESSION_DIR / "cases.json"


def load_cases():
    if not CASES_FILE.exists():
        return []
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", load_cases())
def test_regression_cases(case):
    image_path = REGRESSION_DIR / case["image"]
    expected_path = REGRESSION_DIR / case["expected"]

    if not image_path.exists() or not expected_path.exists():
        pytest.skip("Regression assets not present")

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    image_bytes = image_path.read_bytes()

    result = parse_scorecard_image(image_bytes, debug=False)

    assert result.holes, "No holes parsed"

    for key, expected_value in expected.items():
        if key == "holes":
            for idx, hole_expected in enumerate(expected_value):
                if idx >= len(result.holes):
                    pytest.fail(f"Missing hole {idx + 1}")
                for field, value in hole_expected.items():
                    assert getattr(result.holes[idx], field) == value
        else:
            assert getattr(result, key) == expected_value
