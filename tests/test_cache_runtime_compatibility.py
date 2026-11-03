"""Runtime-only changes retain encoded data; scientific source changes do not."""

import hashlib
from pathlib import Path
import shutil

import pytest

from stmem.data import cache_source


def protected_digest(root):
    """Independent fixture encoding of the documented protected-source ledger."""
    result = hashlib.sha256(b"encoder_cache_protected_sources_v1\0")
    for path in sorted((root / "src").rglob("*.py")):
        name = path.relative_to(root).as_posix()
        if name in cache_source.CACHE_RUNTIME_EXCLUSIONS or name == "src/stmem/data/cache_source.py":
            continue
        if name == "src/stmem/report.py" or name.startswith(("src/stmem/evaluation/", "src/stmem/inspection/")):
            continue
        contents = path.read_bytes()
        result.update(name.encode() + b"\0")
        result.update(len(contents).to_bytes(8, "big"))
        result.update(contents)
    return result.hexdigest()


@pytest.fixture
def audited_tree(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for name, content in {
        "src/stmem/models/spacetime_encoder.py": "# diagnostic encoder source\n",
        "src/stmem/data/canonical.py": "# diagnostic action alignment source\n",
        "src/stmem/train.py": "# diagnostic optimizer execution\n",
        "src/stmem/training/loader.py": "# diagnostic batching execution\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    implementation = root / "src/stmem/data/cache_source.py"
    shutil.copyfile(cache_source.__file__, implementation)
    # This fixture is explicitly synthetic. Bind its protected bytes into a
    # synthetic audit without weakening the production hard-coded baseline.
    monkeypatch.setattr(cache_source, "_AUDITED_PROTECTED_SHA256", protected_digest(root))
    assert cache_source.cache_code_fingerprint(root) == cache_source._AUDITED_LEGACY_SHA256
    return root


@pytest.mark.parametrize("name", sorted(cache_source.CACHE_RUNTIME_EXCLUSIONS))
def test_only_declared_runtime_changes_keep_legacy_identity(audited_tree, name):
    before = cache_source.cache_code_fingerprint(audited_tree)
    path = audited_tree / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# changed runtime consumer, not cached encoder output\n")
    assert cache_source.cache_code_fingerprint(audited_tree) == before
    path.unlink()
    assert cache_source.cache_code_fingerprint(audited_tree) == before


@pytest.mark.parametrize("name", [
    "src/stmem/models/spacetime_encoder.py",
    "src/stmem/data/canonical.py",
    "src/stmem/data/cache.py",
    "src/stmem/config.py",
    "src/stmem/vision/frozen.py",
    "src/stmem/memory/hierarchy.py",
    "src/stmem/training/runtime.py",
    "src/stmem/training/cached_loader_helper.py",
    "src/stmem/training/another_consumer.py",
    "src/stmem/train_extra.py",
    "src/unrecognized_package/encoder.py",
])
def test_changed_or_new_protected_source_never_aliases_old_cache(audited_tree, name):
    path = audited_tree / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# unaudited source changes\n")
    assert cache_source.cache_code_fingerprint(audited_tree) != cache_source._AUDITED_LEGACY_SHA256


def test_missing_or_renamed_encoder_invalidates_cache(audited_tree):
    path = audited_tree / "src/stmem/models/spacetime_encoder.py"
    contents = path.read_bytes()
    path.unlink()
    missing = cache_source.cache_code_fingerprint(audited_tree)
    assert missing != cache_source._AUDITED_LEGACY_SHA256
    path.with_name("renamed_encoder.py").write_bytes(contents)
    renamed = cache_source.cache_code_fingerprint(audited_tree)
    assert renamed != missing and renamed != cache_source._AUDITED_LEGACY_SHA256


def test_changed_fingerprint_implementation_does_not_silently_keep_legacy_cache(audited_tree):
    path = audited_tree / "src/stmem/data/cache_source.py"
    path.write_bytes(path.read_bytes() + b"\n# unaudited fingerprint change\n")
    assert protected_digest(audited_tree) == cache_source._AUDITED_PROTECTED_SHA256
    assert cache_source.cache_code_fingerprint(audited_tree) != cache_source._AUDITED_LEGACY_SHA256


def test_arbitrary_source_tree_is_not_accepted_as_legacy(tmp_path):
    path = tmp_path / "src/stmem/models/encoder.py"
    path.parent.mkdir(parents=True)
    path.write_text("# unrelated source tree\n")
    assert cache_source.cache_code_fingerprint(tmp_path) != cache_source._AUDITED_LEGACY_SHA256


def test_nonencoding_tools_remain_outside_identity(audited_tree):
    for name in ("src/stmem/evaluation/parallel.py", "src/stmem/inspection/report.py", "src/stmem/report.py"):
        path = audited_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# offline reporting only\n")
    assert cache_source.cache_code_fingerprint(audited_tree) == cache_source._AUDITED_LEGACY_SHA256


def test_new_encoding_revision_is_stable_only_across_runtime_edits(audited_tree):
    encoder = audited_tree / "src/stmem/models/spacetime_encoder.py"
    encoder.write_text("# legitimate new encoding revision\n")
    revised = cache_source.cache_code_fingerprint(audited_tree)
    assert revised != cache_source._AUDITED_LEGACY_SHA256
    runtime = audited_tree / "src/stmem/training/cached_loader.py"
    runtime.write_text("# another worker implementation\n")
    assert cache_source.cache_code_fingerprint(audited_tree) == revised
    encoder.write_text("# another encoding revision\n")
    assert cache_source.cache_code_fingerprint(audited_tree) != revised


def test_audited_implementation_digest_matches_its_record():
    root = Path(cache_source.__file__).resolve().parents[3]
    assert cache_source._implementation_fingerprint(root) == cache_source._AUDITED_IMPLEMENTATION_SHA256
