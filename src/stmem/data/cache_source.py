"""Conservative encoded-content identity, with an audited runtime-only migration.

The original format-2 cache covered training execution code as well as its
encoding dependencies. Moving cache *consumption* into workers or building a
derived parent index does not change any encoded leaves. Explicit consumer paths are excluded;
all other data/model/geometry/vision/memory/configuration source remains covered.
Unknown new source files are covered too, rather than assumed safe.

The compatibility record below preserves one known legacy identity only when
every protected source byte still matches its audited baseline. The fingerprint
implementation has its own normalized digest in that record: changing this
module cannot accidentally keep the legacy identity. Its digest excludes only
the digest literal itself, avoiding a circular hash. This is not permission to
accept arbitrary legacy caches or bypass their teacher/data/config checks.

Full repository source provenance is still recorded independently in manifests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re


CACHE_CODE_EXCLUSIONS = (
    "src/stmem/evaluation/",
    "src/stmem/inspection/",
    "src/stmem/report.py",
)

# These modules only consume encoded evidence, during optimization or while
# constructing a separately fingerprinted hierarchy sidecar. They are not
# imported by leaf-cache construction or encoder/preprocessing implementations.
# Keep this an exact path list, not a blanket exclusion of training/ or models/.
CACHE_RUNTIME_EXCLUSIONS = frozenset({
    "src/stmem/train.py",
    "src/stmem/training/loader.py",
    "src/stmem/training/cached_loader.py",
    "src/stmem/training/cached_policy.py",
    "src/stmem/training/packed_context.py",
    "src/stmem/data/hierarchy_cache.py",
    "src/stmem/training/gain_labels.py",
    "src/stmem/training/routers.py",
    "src/stmem/training/label_data.py",
    "src/stmem/training/label_objectives.py",
    "src/stmem/training/label_experiment.py",
})
_SELF_PATH = "src/stmem/data/cache_source.py"

# Encoding sources audited against repository 21d0b24392ce79598149b102da9ce1d785a4c603.
# The 4 November router-objective change additionally excludes the five exact
# label/scorer consumer paths above. None is imported during leaf construction;
# all encoder, alignment, aggregation and policy definitions remain protected.
# The protected digest includes path names, file lengths and complete file bytes
# after only the explicit exclusions above and this separately checked module.
_AUDITED_PROTECTED_SHA256 = "1a9d6cc780fddfd02f0ad41982d951efc817483db92cd2616e956e4f5346c81d"
_AUDITED_LEGACY_SHA256 = "9cddd3ea461824003854f5b0c42595f5f1f975224d4f59ca0be72d0e1e83dc34"
_AUDITED_IMPLEMENTATION_SHA256 = "47c1f6fbecf30660c854ceebed8d5e613836c32094898dd3a0c3573979aef591"


def _implementation_fingerprint(root: Path) -> str:
    path = root / _SELF_PATH
    if not path.is_file():
        return hashlib.sha256(b"missing fingerprint implementation").hexdigest()
    # Only the single top-level expected-hash assignment is normalized. Comments,
    # exclusion sets, compatibility constants and every function remain covered.
    normalized = re.sub(
        rb'^_AUDITED_IMPLEMENTATION_SHA256 = "[0-9a-f]{64}"$',
        b'_AUDITED_IMPLEMENTATION_SHA256 = "' + b"0" * 64 + b'"',
        path.read_bytes(),
        flags=re.MULTILINE,
    )
    return hashlib.sha256(normalized).hexdigest()


def cache_code_fingerprint(root: Path | str | None = None) -> str:
    """Hash protected Python source, preserving exactly the audited old cache.

    ``root`` is the repository root and is injectable for file-change tests.
    Excluded directory names are matched as complete path components, never as
    substrings (for example ``evaluation_utils.py`` remains covered).
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    source = root / "src"
    if not source.is_dir():
        raise ValueError(f"Encoder-cache source root is missing: {source}")
    result = hashlib.sha256(b"encoder_cache_protected_sources_v1\0")
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative in CACHE_RUNTIME_EXCLUSIONS or relative == _SELF_PATH:
            continue
        if relative == CACHE_CODE_EXCLUSIONS[2] or any(
            relative.startswith(prefix) for prefix in CACHE_CODE_EXCLUSIONS[:2]
        ):
            continue
        contents = path.read_bytes()
        result.update(relative.encode("utf-8") + b"\0")
        result.update(len(contents).to_bytes(8, "big"))
        result.update(contents)
    protected = result.hexdigest()
    implementation = _implementation_fingerprint(root)
    if (protected == _AUDITED_PROTECTED_SHA256
            and implementation == _AUDITED_IMPLEMENTATION_SHA256):
        return _AUDITED_LEGACY_SHA256
    # New/changed protected source gets a distinct identity. Runtime-only edits
    # stay irrelevant, but changing encoders, preprocessing, aggregation, action
    # alignment, or this implementation cannot silently reuse the audited cache.
    return hashlib.sha256(
        b"encoder_cache_code_v2\0" + protected.encode() + b"\0" + implementation.encode()
    ).hexdigest()
