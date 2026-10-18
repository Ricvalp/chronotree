"""Task-selective pinned Hugging Face download with SHA256 and safe extraction."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request

TASKS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)
DATASET = "Yinpei/robomme_data_h5"
REVISION = "a5e4e25ffe8af34f64944f9533d06455ce5f8337"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def listing(revision=REVISION):
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{revision}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def safe_extract(archive: Path, destination: Path) -> list[dict]:
    """Streaming extraction rejects links, devices, paths outside root and overwrites."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    members = []
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or member.name.startswith("/"):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"Unsafe archive member type: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if target.stat().st_size != member.size:
                    raise FileExistsError(f"Existing size mismatch: {target}")
                h = hashlib.sha256()
                with tar.extractfile(member) as src:
                    for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                if sha256(target) != h.hexdigest():
                    raise ValueError(f"Existing extracted file failed checksum: {target}")
                members.append({"path": str(target), "bytes": member.size, "sha256": h.hexdigest()})
                continue
            if shutil.disk_usage(destination).free < member.size + 128 * 1024 * 1024:
                raise OSError(f"Insufficient disk space for {member.size} extracted bytes")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".partial")
            h = hashlib.sha256()
            with tar.extractfile(member) as src, tmp.open("wb") as dst:
                for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
                    dst.write(chunk)
            if tmp.stat().st_size != member.size:
                raise OSError("Truncated archive member")
            tmp.replace(target)
            members.append({"path": str(target), "bytes": member.size, "sha256": h.hexdigest()})
    return members


def download_file(url: str, target: Path, expected_size: int, expected_sha: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == expected_size and sha256(target) == expected_sha:
            return
        raise ValueError(f"Existing archive failed checksum: {target}")
    partial = target.with_name(target.name + ".partial")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        resume = start > 0 and response.status == 206
        with partial.open("ab" if resume else "wb") as out:
            shutil.copyfileobj(response, out, 8 * 1024 * 1024)
    if partial.stat().st_size != expected_size or sha256(partial) != expected_sha:
        raise ValueError(f"Download integrity failure: {partial}")
    partial.replace(target)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=(*TASKS, "all"), required=True)
    p.add_argument("--data-root", default=os.getenv("STMEM_DATA_ROOT", "data"))
    p.add_argument("--archive", type=Path, help="Use one already downloaded archive")
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-extract", action="store_true")
    args = p.parse_args(argv)
    tasks = TASKS if "all" in args.tasks else args.tasks
    if args.archive and len(tasks) != 1:
        p.error("--archive requires exactly one task")
    files = {x["path"]: x for x in listing(args.revision)}
    root = Path(args.data_root)
    plan = []
    for task in tasks:
        name = f"record_dataset_{task}.h5.tar.xz"
        f = files[name]
        plan.append({"task": task, "file": name, "bytes": f["size"], "sha256": f["lfs"]["oid"]})
    print(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": args.revision,
                "archives": plan,
                "total_bytes": sum(x["bytes"] for x in plan),
                "extracted_bytes": "measured during extraction; unknown before reading archive",
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    for item in plan:
        archive = args.archive or root / "archives" / item["file"]
        if args.archive:
            if archive.stat().st_size != item["bytes"] or sha256(archive) != item["sha256"]:
                raise ValueError("Local archive failed SHA256/size validation")
        else:
            download_file(
                f"https://huggingface.co/datasets/{DATASET}/resolve/{args.revision}/{item['file']}",
                archive,
                item["bytes"],
                item["sha256"],
            )
        members = [] if args.no_extract else safe_extract(archive, root / "raw")
        manifest = {
            **item,
            "revision": args.revision,
            "members": members,
            "extracted_bytes": sum(x["bytes"] for x in members),
        }
        path = root / "manifests" / f"{item['task']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
