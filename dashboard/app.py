from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import load_wav
from src.features import extract_mfcc_features

DATA_ROOT = ROOT / "data" / "raw_timit" / "data"
TEST_ROOT = DATA_ROOT / "TEST"
LIST_FILE = ROOT / "data" / "lists" / "test_enrollment_list.txt"
UBM_PATH = ROOT / "results" / "models" / "ubm_model.pkl"
ENROLLED_MODELS_DIR = ROOT / "results" / "enrolled_models"
INDEX_HTML = Path(__file__).with_name("index.html")


app = FastAPI(title="Speaker Verification Dashboard", version="1.0.0")


class VerifyRequest(BaseModel):
    target_speaker: str = Field(..., description="Enrolled speaker model to score against")
    trial_path: str = Field(..., description="Relative path to a wav file under the repository root")
    threshold: float = Field(0.0, description="Decision threshold for verification")


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Path is outside the repository root: {path}")
    return resolved.relative_to(ROOT).as_posix()


def _resolve_repo_path(relative_path: str) -> Path:
    candidate = (ROOT / Path(relative_path)).resolve()
    if not candidate.is_relative_to(ROOT):
        raise HTTPException(status_code=400, detail="Path must stay inside the repository root")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {relative_path}")
    return candidate


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32)


def _score_model(model: Any, features: np.ndarray) -> float:
    if hasattr(model, "score"):
        return float(model.score(features))
    if hasattr(model, "gmm") and hasattr(model.gmm, "score"):
        return float(model.gmm.score(features))
    raise TypeError(f"Unsupported model type: {type(model)!r}")


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    files_by_speaker: dict[str, list[str]] = {}

    if LIST_FILE.exists():
        for raw_line in LIST_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or "|" not in line:
                continue
            speaker_ref, paths_str = line.split("|", 1)
            speaker_id = Path(speaker_ref).name
            speaker_files = files_by_speaker.setdefault(speaker_id, [])
            for raw_path in paths_str.split(","):
                wav_path = Path(raw_path.strip())
                if not wav_path.exists():
                    continue
                try:
                    speaker_files.append(_repo_relative(wav_path))
                except ValueError:
                    continue
    else:
        for wav_path in TEST_ROOT.rglob("*.WAV.wav"):
            if wav_path.name.startswith("SA"):
                continue
            speaker_id = wav_path.parent.name
            files_by_speaker.setdefault(speaker_id, []).append(_repo_relative(wav_path))

    for speaker_id, paths in files_by_speaker.items():
        deduped = list(dict.fromkeys(paths))
        deduped.sort()
        files_by_speaker[speaker_id] = deduped

    target_speakers = sorted(
        speaker_id
        for speaker_id in files_by_speaker
        if (ENROLLED_MODELS_DIR / f"speaker_{speaker_id}.pkl").exists()
    )
    trial_speakers = sorted(files_by_speaker)

    default_target = target_speakers[0] if target_speakers else ""
    default_trial = default_target if default_target in files_by_speaker else (trial_speakers[0] if trial_speakers else "")

    return {
        "targetSpeakers": [
            {
                "id": speaker_id,
                "label": speaker_id,
                "fileCount": len(files_by_speaker[speaker_id]),
            }
            for speaker_id in target_speakers
        ],
        "trialSpeakers": [
            {
                "id": speaker_id,
                "label": speaker_id,
                "fileCount": len(files_by_speaker[speaker_id]),
            }
            for speaker_id in trial_speakers
        ],
        "filesBySpeaker": files_by_speaker,
        "defaultTargetSpeaker": default_target,
        "defaultTrialSpeaker": default_trial,
        "defaultThreshold": 0.0,
        "ubmReady": UBM_PATH.exists(),
        "modelCount": len(target_speakers),
        "speakerCount": len(trial_speakers),
    }


@lru_cache(maxsize=1)
def load_ubm() -> Any:
    if not UBM_PATH.exists():
        raise FileNotFoundError(f"Missing UBM model: {UBM_PATH}")
    return joblib.load(UBM_PATH)


@lru_cache(maxsize=128)
def load_target_model(target_speaker: str) -> Any:
    model_path = ENROLLED_MODELS_DIR / f"speaker_{target_speaker}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing enrolled model: {model_path}")
    return joblib.load(model_path)


def score_gmm_trial(target_speaker: str, trial_path: str) -> dict[str, Any]:
    ubm = load_ubm()
    target_model = load_target_model(target_speaker)

    wav_path = _resolve_repo_path(trial_path)
    audio, sr = load_wav(wav_path)
    audio = _normalize_audio(np.asarray(audio))
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="The selected audio file is empty")

    trimmed_audio, _ = librosa.effects.trim(audio, top_db=25)
    if trimmed_audio.size > 0:
        audio = trimmed_audio

    features = extract_mfcc_features(audio, sr)
    if features.shape[0] < 3:
        raise HTTPException(status_code=400, detail="The selected file is too short for verification")

    target_score = _score_model(target_model, features)
    ubm_score = _score_model(ubm, features)
    llr = target_score - ubm_score
    trial_speaker = Path(trial_path).parent.name
    verified = llr >= 0.0

    return {
        "targetSpeaker": target_speaker,
        "trialSpeaker": trial_speaker,
        "trialPath": trial_path,
        "verified": verified,
        "score": llr,
        "threshold": 0.0,
        "targetLogLikelihood": target_score,
        "ubmLogLikelihood": ubm_score,
        "durationSec": float(len(audio) / sr),
        "frameCount": int(features.shape[0]),
        "decisionRule": "llr >= threshold",
        "modelPath": str((ENROLLED_MODELS_DIR / f"speaker_{target_speaker}.pkl").relative_to(ROOT).as_posix()),
    }


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/catalog")
def api_catalog() -> JSONResponse:
    return JSONResponse(load_catalog())


@app.get("/audio")
def audio(path: str = Query(..., description="Relative path to a wav file")) -> FileResponse:
    wav_path = _resolve_repo_path(path)
    return FileResponse(wav_path, media_type="audio/wav", filename=wav_path.name)


@app.post("/api/verify")
def api_verify(payload: VerifyRequest) -> JSONResponse:
    catalog = load_catalog()
    if payload.target_speaker not in catalog["filesBySpeaker"]:
        raise HTTPException(status_code=404, detail=f"Unknown speaker: {payload.target_speaker}")
    if not (ENROLLED_MODELS_DIR / f"speaker_{payload.target_speaker}.pkl").exists():
        raise HTTPException(status_code=404, detail=f"No enrolled model found for speaker: {payload.target_speaker}")

    result = score_gmm_trial(payload.target_speaker, payload.trial_path)
    result["threshold"] = float(payload.threshold)
    result["verified"] = result["score"] >= payload.threshold
    result["decision"] = "verified" if result["verified"] else "not_verified"
    result["groundTruth"] = "genuine" if result["trialSpeaker"] == result["targetSpeaker"] else "impostor"
    result["decisionRule"] = "score >= threshold"
    return JSONResponse(result)


def main() -> None:
    import uvicorn

    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()