"""Temporary media handling.

Videos are NEVER stored permanently. They are streamed into the system temp
directory, used for analysis, then deleted — guaranteed via a context manager
that cleans up even on exceptions.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Iterator

import requests

from core.logging_config import get_logger

logger = get_logger("storage")

_TEMP_PREFIX = "adintel_"
_MAX_BYTES = 64 * 1024 * 1024  # 64 MB safety cap


@contextlib.contextmanager
def temporary_download(url: str, suffix: str = ".mp4", timeout: int = 30) -> Iterator[Path | None]:
    """Download `url` to a temp file, yield its path, always delete afterwards.

    Yields None if the download fails — callers must handle the None case and
    fall back to metadata-only analysis (never crash).
    """
    fd, tmp_name = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=suffix)
    os.close(fd)
    path = Path(tmp_name)
    downloaded = False
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            written = 0
            with open(path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > _MAX_BYTES:
                        logger.warning("Video exceeds %s bytes; aborting download", _MAX_BYTES)
                        break
                    fh.write(chunk)
            downloaded = written > 0
        yield path if downloaded else None
    except Exception as exc:
        logger.warning("Video download failed (%s): %s", url[:80] if url else "", exc)
        yield None
    finally:
        cleanup(path)


def cleanup(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
        if path and path.exists():
            path.unlink()
            logger.debug("Deleted temp file %s", path)


def sweep_orphans() -> int:
    """Delete any stale adintel_* temp files left by crashed runs."""
    removed = 0
    tmp = Path(tempfile.gettempdir())
    for p in tmp.glob(f"{_TEMP_PREFIX}*"):
        with contextlib.suppress(OSError):
            p.unlink()
            removed += 1
    if removed:
        logger.info("Swept %s orphan temp files", removed)
    return removed
