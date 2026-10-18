"""Shared location rules for prepared data and its alignment evidence."""

from pathlib import Path


def alignment_path(config: dict) -> Path:
    """Resolve explicit evidence paths while relocating the default with data_root.

    Relative custom paths retain the repository-relative convention used by the
    other project paths. The optional {data_root} template is useful for sites.
    """
    root = Path(config.get("paths", {}).get("data_root", "data"))
    configured = config.get("data", {}).get("alignment_path")
    if configured is None or Path(configured) == Path("data/alignment.json"):
        return root / "alignment.json"
    return Path(str(configured).format(data_root=str(root))).expanduser()
