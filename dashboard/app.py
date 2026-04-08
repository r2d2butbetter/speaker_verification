from __future__ import annotations

import importlib.util
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
LSTM_DIR = ROOT / "results" / "models" / "lstm"
INDEX_HTML = Path(__file__).with_name("index.html")


app = FastAPI(title="Speaker Verification Dashboard", version="1.0.0")


class VerifyRequest(BaseModel):
    model_type: str = Field("ubm", description="Verification backend to use: ubm or bilstm")
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


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _epoch_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


@lru_cache(maxsize=1)
def _find_lstm_checkpoint() -> Path | None:
    candidates = [
        LSTM_DIR / "lstm_best.pt",
        LSTM_DIR / "lstm_final.pt",
    ]
    candidates.extend(sorted(LSTM_DIR.glob("lstm_epoch_*.pt"), key=_epoch_number, reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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

    lstm_checkpoint = _find_lstm_checkpoint()
    model_families = [
        {
            "id": "ubm",
            "label": "UBM-GMM",
            "description": "Log-likelihood ratio against the universal background model.",
            "available": UBM_PATH.exists(),
        },
        {
            "id": "bilstm",
            "label": "BiLSTM",
            "description": "Cosine similarity between d-vectors from the BiLSTM encoder.",
            "available": bool(lstm_checkpoint) and _torch_available(),
        },
    ]

    default_model_type = next((item["id"] for item in model_families if item["available"]), "ubm")

    return {
        "modelFamilies": model_families,
        "defaultModelType": default_model_type,
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
        "bilstmReady": bool(lstm_checkpoint) and _torch_available(),
        "modelCount": len(model_families),
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


@lru_cache(maxsize=1)
def _load_lstm_backend() -> tuple[Any, Any, Path]:
    if not _torch_available():
        raise HTTPException(status_code=503, detail="BiLSTM verification requires torch, which is not installed in this environment.")

    checkpoint_path = _find_lstm_checkpoint()
    if checkpoint_path is None:
        raise HTTPException(status_code=503, detail="BiLSTM checkpoint not found under results/models/lstm/")

    import torch

    from lstm.model import SpeakerLSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = SpeakerLSTM(input_dim=39, hidden_dim=256, num_layers=2, embedding_dim=128, num_classes=None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, device, checkpoint_path


def _audio_to_lstm_embedding(model: Any, device: Any, audio: np.ndarray, sr: int, audio_label: str) -> tuple[np.ndarray, int]:
    import torch

    from src.features import mfcc_with_deltas

    audio = _normalize_audio(np.asarray(audio))
    if audio.size == 0:
        raise HTTPException(status_code=400, detail=f"The selected audio file is empty: {audio_label}")

    feats = mfcc_with_deltas(audio.astype(np.float32), sr)
    frame_count = int(feats.shape[1])
    if frame_count < 3:
        raise HTTPException(status_code=400, detail=f"The selected file is too short for BiLSTM verification: {audio_label}")

    feats_t = torch.tensor(feats.T, dtype=torch.float32).unsqueeze(0).to(device)
    mean = feats_t.mean(dim=1, keepdim=True)
    std = feats_t.std(dim=1, keepdim=True) + 1e-6
    feats_t = (feats_t - mean) / std

    with torch.no_grad():
        embedding = model(feats_t).cpu().numpy().flatten()

    return embedding, frame_count


def _build_lstm_template(model: Any, device: Any, target_speaker: str) -> tuple[np.ndarray, int]:
    catalog = load_catalog()
    enrollment_paths = catalog["filesBySpeaker"].get(target_speaker, [])[:3]
    if not enrollment_paths:
        raise HTTPException(status_code=404, detail=f"No enrollment files found for speaker: {target_speaker}")

    embeddings = []
    frame_total = 0
    for relative_path in enrollment_paths:
        wav_path = _resolve_repo_path(relative_path)
        audio, sr = load_wav(wav_path)
        embedding, frame_count = _audio_to_lstm_embedding(model, device, np.asarray(audio), sr, wav_path.name)
        embeddings.append(embedding)
        frame_total += frame_count

    template = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(template)
    if norm <= 0:
        raise HTTPException(status_code=500, detail=f"Failed to build BiLSTM template for speaker: {target_speaker}")

    return template / norm, frame_total


@lru_cache(maxsize=128)
def load_lstm_template(target_speaker: str) -> tuple[np.ndarray, Path, int]:
    model, device, checkpoint_path = _load_lstm_backend()
    template, frame_total = _build_lstm_template(model, device, target_speaker)
    return template, checkpoint_path, frame_total


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
        "modelType": "ubm",
        "modelLabel": "UBM-GMM",
        "scoreType": "llr",
    }


def score_bilstm_trial(target_speaker: str, trial_path: str) -> dict[str, Any]:
    wav_path = _resolve_repo_path(trial_path)
    model, device, checkpoint_path = _load_lstm_backend()
    target_template, _, enrollment_frame_total = load_lstm_template(target_speaker)

    audio, sr = load_wav(wav_path)
    audio = _normalize_audio(np.asarray(audio))
    duration_sec = float(len(audio) / sr) if sr else 0.0
    trial_embedding, frame_count = _audio_to_lstm_embedding(model, device, audio, sr, wav_path.name)
    trial_norm = np.linalg.norm(trial_embedding)
    if trial_norm <= 0:
        raise HTTPException(status_code=400, detail="Failed to compute a valid BiLSTM embedding for the selected file")

    trial_embedding = trial_embedding / trial_norm
    score = float(np.dot(trial_embedding, target_template))
    trial_speaker = Path(trial_path).parent.name
    catalog = load_catalog()
    enrollment_count = len(catalog["filesBySpeaker"].get(target_speaker, [])[:3])

    return {
        "targetSpeaker": target_speaker,
        "trialSpeaker": trial_speaker,
        "trialPath": trial_path,
        "verified": score >= 0.0,
        "score": score,
        "threshold": 0.0,
        "targetLogLikelihood": None,
        "ubmLogLikelihood": None,
        "durationSec": duration_sec,
        "frameCount": frame_count,
        "decisionRule": "cosine >= threshold",
        "modelPath": str(checkpoint_path.relative_to(ROOT).as_posix()),
        "modelType": "bilstm",
        "modelLabel": "BiLSTM",
        "scoreType": "cosine",
        "enrollmentCount": enrollment_count,
        "enrollmentFrameCount": enrollment_frame_total,
        "checkpointPath": str(checkpoint_path.relative_to(ROOT).as_posix()),
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

    model_type = payload.model_type.strip().lower()
    if model_type == "ubm":
        if not (ENROLLED_MODELS_DIR / f"speaker_{payload.target_speaker}.pkl").exists():
            raise HTTPException(status_code=404, detail=f"No enrolled model found for speaker: {payload.target_speaker}")
        result = score_gmm_trial(payload.target_speaker, payload.trial_path)
    elif model_type == "bilstm":
        result = score_bilstm_trial(payload.target_speaker, payload.trial_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {payload.model_type}")

    result["threshold"] = float(payload.threshold)
    result["verified"] = result["score"] >= payload.threshold
    result["decision"] = "verified" if result["verified"] else "not_verified"
    result["groundTruth"] = "genuine" if result["trialSpeaker"] == result["targetSpeaker"] else "impostor"
    result["modelType"] = model_type
    return JSONResponse(result)


def main() -> None:
    import uvicorn

    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()