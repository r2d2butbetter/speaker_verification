"""
Data I/O utilities for TIMIT and experiment splits.

- Reads TIMIT (converted WAVs) and metadata
- Enforces strict train/enroll/test splits from text lists
"""

from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf


def load_wav(path: str | Path) -> Tuple[list, int]:
    """Load a WAV file.

    Returns (audio, sr). Minimal placeholder to unblock scaffolding.
    """
    audio, sr = sf.read(str(path), dtype="float32")
    return audio, sr


def read_list(file_path: str | Path) -> List[str]:
    """Read a newline-delimited list file, ignoring blanks and comments."""
    p = Path(file_path)
    if not p.exists():
        return []
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def get_splits(lists_dir: str | Path) -> Dict[str, List[str]]:
    """Return dict with 'background', 'enroll', 'test' speaker or file IDs."""
    lists_dir = Path(lists_dir)
    return {
        "background": read_list(lists_dir / "background.txt"),
        "enroll": read_list(lists_dir / "enroll.txt"),
        "test": read_list(lists_dir / "test.txt"),
    }


def find_wavs(root: str | Path, pattern: str = "**/*.wav") -> List[Path]:
    """Recursively find WAV files under root.

    Placeholder for later filtering to specific TIMIT subsets.
    """
    return sorted(Path(root).glob(pattern))
