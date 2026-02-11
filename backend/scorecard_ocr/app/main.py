from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path
import logging
import time

from app.models import ScorecardParseResponse
from app.pipeline import parse_scorecard_image

load_dotenv()
repo_env = Path(__file__).resolve().parents[2] / ".env"
if repo_env.exists():
    load_dotenv(repo_env)

app = FastAPI(title="GolfSum Scorecard OCR", version="0.1.0")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app.main")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/scorecard/parse", response_model=ScorecardParseResponse)
async def parse_scorecard(
    image: UploadFile = File(...),
    debug: bool = Query(False),
    mode: str = Query("course"),
) -> ScorecardParseResponse:
    if not image.content_type or "image" not in image.content_type:
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image payload")

    start = time.time()
    logger.info(
        "OCR request received filename=%s content_type=%s bytes=%s mode=%s debug=%s",
        image.filename,
        image.content_type,
        len(content),
        mode,
        debug,
    )
    result = parse_scorecard_image(content, debug=debug, mode=mode)
    elapsed = (time.time() - start) * 1000
    logger.info(
        "OCR response ready mode=%s confidence=%s flags=%s elapsed_ms=%.0f",
        mode,
        result.get("confidence"),
        result.get("flags"),
        elapsed,
    )
    return JSONResponse(content=result)
