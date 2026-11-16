#!/usr/bin/env python3
"""Read-only I/O adapter for the original utility study's case planner.

Run this file with the archived study's ``src`` on PYTHONPATH. Its scientific
module and ``load_rows`` function remain unchanged. Only that module's NumPy
binding is proxied so a compressed NPZ column is decoded once per open file,
instead of once per row. Checksums, order, scalar conversion and case selection
remain the archived implementation's responsibility.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial, wraps
import json
from pathlib import Path
import sys
import time


ADAPTER_SCHEMA = "refinement_generated_npz_columns_v1"


class CachedColumns:
    """Forward an NpzFile while retaining each requested column until close."""

    def __init__(self, archive):
        self._archive = archive
        self._columns = {}

    def __getattr__(self, name):
        return getattr(self._archive, name)

    def __getitem__(self, key):
        if key not in self._columns:
            self._columns[key] = self._archive[key]
        return self._columns[key]

    def __enter__(self):
        self._archive.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self._archive.__exit__(*args)
        finally:
            self._columns.clear()

    def close(self):
        try:
            return self._archive.close()
        finally:
            self._columns.clear()


class ColumnCachingNumpy:
    """A local forwarding binding; the process-wide numpy module is untouched."""

    def __init__(self, numpy_module):
        self._numpy = numpy_module

    def __getattr__(self, name):
        return getattr(self._numpy, name)

    def load(self, *args, **kwargs):
        result = self._numpy.load(*args, **kwargs)
        if isinstance(result, self._numpy.lib.npyio.NpzFile):
            return CachedColumns(result)
        return result


def progress(phase, *, progress_file=None, **fields):
    line = json.dumps({"event": "generated_plan_progress", "phase": phase, **fields})
    print(line, file=sys.stderr, flush=True)
    if progress_file is not None:
        path = Path(progress_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(line + "\n")


@contextmanager
def cached_row_loading(module, *, emit=progress, every=8):
    """Temporarily optimize only this module's reads and expose progress."""
    if every < 1:
        raise ValueError("The progress interval must be positive")
    original_numpy, original_load_rows = module.np, module.load_rows
    started = time.monotonic()
    count = rows_read = 0
    status = "failed"

    @wraps(original_load_rows)
    def load_rows(path):
        nonlocal count, rows_read
        count += 1
        report = count == 1 or count % every == 0
        if report:
            emit("rows_started", file_number=count, path=str(path),
                 elapsed_seconds=time.monotonic() - started)
        rows, metadata = original_load_rows(path)
        rows_read += len(rows)
        if report:
            emit("rows_completed", file_number=count, path=str(path), rows=len(rows),
                 total_rows=rows_read, elapsed_seconds=time.monotonic() - started)
        return rows, metadata

    module.np = ColumnCachingNumpy(original_numpy)
    module.load_rows = load_rows
    try:
        emit("rows_loading_started", adapter=ADAPTER_SCHEMA)
        yield
        status = "complete"
    finally:
        module.np, module.load_rows = original_numpy, original_load_rows
        emit("rows_loading_finished", status=status, files=count, rows=rows_read,
             elapsed_seconds=time.monotonic() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--progress-file", type=Path)
    args, scientific_args = parser.parse_known_args(argv)
    emit = partial(progress, progress_file=args.progress_file)
    started = time.monotonic()
    emit("planner_import_started", adapter=ADAPTER_SCHEMA)
    # Do not add the current checkout to sys.path: these imports must resolve
    # through the launcher's explicitly selected original study source.
    from stmem.inspection import refinement_generated, refinement_utility

    emit("planner_started", scientific_module=refinement_generated.__file__,
         elapsed_seconds=time.monotonic() - started)
    try:
        with cached_row_loading(refinement_utility, emit=emit):
            result = refinement_generated.main(scientific_args)
    except BaseException:
        emit("planner_failed", elapsed_seconds=time.monotonic() - started)
        raise
    emit("planner_finished", elapsed_seconds=time.monotonic() - started)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
