"""Conservative code identity for cached learned observation encodings.

Evaluation orchestration, local inspection tools and report rendering do not
produce the training cache's encoded content. Their implementation can evolve
without discarding an expensive encoder cache. Every other Python source file
under src/ remains covered, including data, models, training, geometry, memory,
configuration, vision and this fingerprint implementation itself. In particular,
this is deliberately broader than an inferred Python import dependency graph.

The complete repository source_revision is recorded separately in the final
cache manifest for audit; this narrower identity determines compatibility.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


CACHE_CODE_EXCLUSIONS = (
    "src/stmem/evaluation/",
    "src/stmem/inspection/",
    "src/stmem/report.py",
)


def cache_code_fingerprint(root: Path | str | None = None) -> str:
    """Hash all Python source except the three explicitly non-encoding paths.

    ``root`` is the repository root and is injectable for file-change tests.
    Excluded directory names are matched as complete path components, never as
    substrings (for example ``evaluation_utils.py`` remains covered).
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    source = root / "src"
    if not source.is_dir():
        raise ValueError(f"Encoder-cache source root is missing: {source}")
    result = hashlib.sha256(b"encoder_cache_code_v1\0")
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative == CACHE_CODE_EXCLUSIONS[2] or any(
            relative.startswith(prefix) for prefix in CACHE_CODE_EXCLUSIONS[:2]
        ):
            continue
        contents = path.read_bytes()
        result.update(relative.encode("utf-8") + b"\0")
        result.update(len(contents).to_bytes(8, "big"))
        result.update(contents)
    return result.hexdigest()
