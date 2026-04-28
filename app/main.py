import logging
import time
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from .prior_relevance import predict_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("relevant-priors-api")

app = FastAPI(title="Relevant Priors API", version="1.0.0")


class Study(BaseModel):
    study_id: str
    study_description: str = ""
    study_date: str = ""


class Case(BaseModel):
    case_id: str
    patient_id: str | None = None
    patient_name: str | None = None
    current_study: Study
    prior_studies: List[Study] = Field(default_factory=list)


class PredictRequest(BaseModel):
    challenge_id: str | None = None
    schema_version: int | None = None
    generated_at: str | None = None
    cases: List[Case] = Field(default_factory=list)


@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "message": "POST /predict"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(payload: PredictRequest, request: Request) -> Dict[str, Any]:
    started = time.time()
    request_id = request.headers.get("x-request-id", "-")
    cases_as_dict = [case.model_dump() for case in payload.cases]
    prior_count = sum(len(case.get("prior_studies", [])) for case in cases_as_dict)

    logger.info(
        "request_id=%s cases=%s priors=%s challenge_id=%s",
        request_id,
        len(cases_as_dict),
        prior_count,
        payload.challenge_id,
    )

    predictions = predict_cases(cases_as_dict)

    elapsed_ms = round((time.time() - started) * 1000, 2)
    logger.info(
        "request_id=%s predictions=%s elapsed_ms=%s",
        request_id,
        len(predictions),
        elapsed_ms,
    )
    return {"predictions": predictions}
