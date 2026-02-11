from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, Optional
import logging

import httpx


GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


COURSE_PROMPT = """You are a golf scorecard parser. Extract structured data from the image.
Return ONLY valid JSON (no markdown, no commentary) with this schema:
{
  "par": [18 integers or nulls],
  "handicapMen": [18 integers or nulls],
  "handicapWomen": [18 integers or nulls],
  "teeBoxes": [
    {
      "name": "string",
      "yardages": [18 integers or nulls],
      "ratingMen": number or null,
      "slopeMen": integer or null,
      "ratingWomen": number or null,
      "slopeWomen": integer or null
    }
  ]
}

Rules:
- Use hole order 1..18. If only 9 holes are visible, fill the other 9 with nulls.
- Yardages should be 80-700 when present.
- Par values should be 3-5 when present.
- Handicap values should be 1-18 when present.
- Tee box names should be the printed names (e.g., CHAMPIONSHIP, BLACK, WHITE, WILDFLOWER).
- Only include rating/slope if they are explicitly printed. Otherwise use null.
- If a value is not present or unreadable, use null.
"""

COMPLETED_ROUND_PROMPT = """You are a golf scorecard parser. Extract structured data from the image.
Return ONLY valid JSON (no markdown, no commentary) with this schema:
{
  "par": [18 integers or nulls],
  "handicapMen": [18 integers or nulls],
  "handicapWomen": [18 integers or nulls],
  "teeBoxes": [
    {
      "name": "string",
      "yardages": [18 integers or nulls],
      "ratingMen": number or null,
      "slopeMen": integer or null,
      "ratingWomen": number or null,
      "slopeWomen": integer or null
    }
  ],
  "player": {
    "name": "string or null",
    "date": "string or null",
    "holes": [
      {
        "hole": 1,
        "score": integer or null,
        "putts": integer or null,
        "fairway": true/false/null,
        "green": true/false/null,
        "upDown": true/false/null,
        "penalties": integer or null
      }
    ]
  }
}

Rules:
- Use hole order 1..18. If only 9 holes are visible, fill the other 9 with nulls.
- Yardages should be 80-700 when present.
- Par values should be 3-5 when present.
- Handicap values should be 1-18 when present.
- Tee box names should be the printed names (e.g., CHAMPIONSHIP, BLACK, WHITE, WILDFLOWER).
- Only include rating/slope if they are explicitly printed. Otherwise use null.
- If a value is not present or unreadable, use null.
- For fairway/green/upDown: use true/false when a checkmark/X is clearly visible, otherwise null.
- If multiple numeric rows could be scores (e.g., rows labeled D/P), only treat the row aligned to the HOLE numbers as scores. If ambiguous, set score to null.
"""

PLAYER_ONLY_PROMPT = """You are a golf scorecard parser focused on PLAYER results. Extract ONLY the player's handwritten data from the image.
Return ONLY valid JSON (no markdown, no commentary) with this schema:
{
  "player": {
    "name": "string or null",
    "date": "string or null",
    "holes": [
      {
        "hole": 1,
        "score": integer or null,
        "putts": integer or null,
        "fairway": true/false/null,
        "green": true/false/null,
        "upDown": true/false/null,
        "penalties": integer or null
      }
    ]
  }
}

Rules:
- Always include holes 1..18 in order. If a value is missing, use null.
- Scores are handwritten in a row labeled by player name or left unlabeled near the middle.
- Putts may be labeled PUTTS, PTS, or noted as a row of numbers.
- Fairways and greens are often checkmarks/Xs/lines; use true/false when clear, else null.
- If you cannot find the player name or date, return null for those fields.
"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: extract first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_scorecard_with_gemini(image_bytes: bytes, mode: str = "course") -> Dict[str, Any]:
    logger = logging.getLogger("app.gemini")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("missing_gemini_api_key")
        return {
            "error": "missing_gemini_api_key",
            "message": "Set GEMINI_API_KEY in the environment.",
        }

    if mode not in {"course", "completed", "player"}:
        return {
            "error": "invalid_mode",
            "message": f"Unsupported mode: {mode}",
        }

    if mode == "completed":
        prompt = COMPLETED_ROUND_PROMPT
    elif mode == "player":
        prompt = PLAYER_ONLY_PROMPT
    else:
        prompt = COURSE_PROMPT
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{GEMINI_ENDPOINT}?key={api_key}",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("gemini_http_error status=%s body=%s", exc.response.status_code, exc.response.text)
        return {
            "error": "gemini_http_error",
            "status_code": exc.response.status_code,
            "body": exc.response.text,
        }
    except httpx.RequestError as exc:
        logger.error("gemini_request_error message=%s", str(exc))
        return {
            "error": "gemini_request_error",
            "message": str(exc),
        }

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        logger.error("gemini_response_parse_failed")
        return {
            "error": "gemini_response_parse_failed",
            "raw": data,
        }

    parsed = _extract_json(text)
    if not parsed:
        logger.error("gemini_json_parse_failed")
        return {
            "error": "gemini_json_parse_failed",
            "raw_text": text,
        }
    return parsed
