"""One causal block construction shared by offline teacher/caches and online reads."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import torch

from .hierarchy import Archive, maximum_root_count, stable_id
from .records import MemoryRecord


def encode_blocks(
    conditioning: Sequence[Any], execution_prefix: Sequence[Any], encoder: Any, block_size: int | None = None
) -> Iterator[tuple[str, str, list[MemoryRecord], bool]]:
    """Execution prefix MUST exclude the separately supplied current observation."""
    block_size = block_size or encoder.block_size
    if block_size != encoder.block_size:
        raise ValueError("Block construction and encoder block lengths must agree")
    seen: set[tuple[str, str]] = set()
    for source, frames in (("conditioning", conditioning), ("execution", execution_prefix)):
        if any(f.source != source for f in frames):
            raise ValueError("Source bank contains frames from another source")
        for i, frame in enumerate(frames):
            identity = (source, frame.event_id)
            if identity in seen:
                raise ValueError("Duplicate observation in offline causal prefix")
            seen.add(identity)
            if i and frame.time <= frames[i - 1].time:
                raise ValueError("Source-local frame clocks must increase strictly")
        for start in range(0, len(frames), block_size):
            block = frames[start : start + block_size]
            completed = source == "conditioning" or len(block) == block_size
            block_id = stable_id(
                source + (":complete" if completed else ":prefix"), (f.event_id for f in block)
            )
            yield (
                source,
                block_id,
                encoder.encode_frames(block, source, block_id, completed=completed),
                completed,
            )


def build_archive(
    conditioning: Sequence[Any], execution_prefix: Sequence[Any], encoder: Any, block_size: int | None = None
) -> Archive:
    archive = Archive()
    for source, block_id, leaves, completed in encode_blocks(
        conditioning, execution_prefix, encoder, block_size
    ):
        if completed:
            archive.add_block(leaves, block_id)
        else:
            archive.set_pending(source, leaves)
    return archive


class StreamingMemory:
    """Frozen archive: CPU storage, replaceable partial blocks, current exactly once.

    An observe call promotes the previous current observation into history. A new
    frame remains outside history until a later observation arrives. Repeated
    event identities are idempotent, including retried policy-service calls.
    """

    def __init__(
        self,
        encoder: Any,
        block_size: int | None = None,
        *,
        archive_device: str = "cpu",
        max_execution_frames: int = 1400,
        max_conditioning_frames: int = 1400,
    ):
        self.encoder = encoder
        self.block_size = block_size or encoder.block_size
        if self.block_size != encoder.block_size:
            raise ValueError("Streaming and encoder block lengths disagree")
        self.archive_device = archive_device
        self.max_execution_frames, self.max_conditioning_frames = (
            max_execution_frames,
            max_conditioning_frames,
        )
        self._archive = Archive()
        self.current: Any | None = None
        self._pending_frames: list[Any] = []
        self._seen: set[tuple[str, str]] = set()
        self._prefix_dirty = False
        self.execution_observations = 0
        self.encoding_calls = 0
        self.encoded_frames = 0

    def validate_budget(self, budget: int) -> None:
        bound = maximum_root_count(self.max_execution_frames, self.max_conditioning_frames, self.block_size)
        if budget < bound:
            raise ValueError(
                f"Budget {budget} is below worst supported root bound {bound}; "
                "raise the budget or explicitly shorten the supported episode profile"
            )

    def _encode(self, frames: Sequence[Any], source: str, completed: bool) -> tuple[str, list[MemoryRecord]]:
        block_id = stable_id(source + (":complete" if completed else ":prefix"), (f.event_id for f in frames))
        with torch.no_grad():
            leaves = self.encoder.encode_frames(frames, source, block_id, completed=completed)
        self.encoding_calls += 1
        self.encoded_frames += len(frames)
        return block_id, [r.to(self.archive_device, detach=True) for r in leaves]

    def reset(self, conditioning: Sequence[Any] = (), current: Any | None = None) -> None:
        self._archive.clear()
        self._pending_frames.clear()
        self._seen.clear()
        self.current = None
        self.execution_observations = self.encoding_calls = self.encoded_frames = 0
        self._prefix_dirty = False
        if len(conditioning) > self.max_conditioning_frames:
            raise ValueError("Conditioning video exceeds the declared supported maximum")
        for i, frame in enumerate(conditioning):
            key = (frame.source, frame.event_id)
            if frame.source != "conditioning" or key in self._seen:
                raise ValueError("Reset conditioning frames must have unique ordered conditioning identities")
            if i and frame.time <= conditioning[i - 1].time:
                raise ValueError("Conditioning source-local times must increase")
            self._seen.add(key)
        for start in range(0, len(conditioning), self.block_size):
            block_id, leaves = self._encode(
                conditioning[start : start + self.block_size], "conditioning", True
            )
            self._archive.add_block(leaves, block_id)
        if current is not None:
            self.observe(current)

    def observe(self, frame: Any) -> bool:
        key = (frame.source, frame.event_id)
        if key in self._seen:
            return False
        if frame.source != "execution":
            raise ValueError("Conditioning must be installed once through reset")
        if self.execution_observations >= self.max_execution_frames + 1:
            raise ValueError("Execution history exceeds the declared supported maximum")
        if self.current is not None:
            if frame.time <= self.current.time:
                raise ValueError("New execution frames must have increasing source-local timestamps")
            self._pending_frames.append(self.current)
            self._prefix_dirty = True
            if len(self._pending_frames) == self.block_size:
                block_id, leaves = self._encode(self._pending_frames, "execution", True)
                self._archive.add_block(leaves, block_id)
                self._pending_frames.clear()
                self._prefix_dirty = False
        self.current = frame
        self.execution_observations += 1
        self._seen.add(key)
        return True

    @property
    def archive(self) -> Archive:
        if self._prefix_dirty:
            _, leaves = self._encode(self._pending_frames, "execution", False)
            self._archive.set_pending("execution", leaves)
            self._prefix_dirty = False
        return self._archive

    @property
    def roots(self) -> list[MemoryRecord]:
        return self.archive.roots

    @property
    def leaves(self) -> list[MemoryRecord]:
        return self.archive.leaves

    @property
    def storage(self) -> dict[str, int]:
        pending_bytes = 0
        for frame in self._pending_frames:
            pending_bytes += sum(
                x.numel() * x.element_size()
                for x in (frame.points, frame.valid, frame.proprio, frame.previous_action)
            )
        identity_bytes = sum(len(s.encode()) + len(i.encode()) for s, i in self._seen)
        return {
            "archive_payload_bytes": self.archive.storage_bytes,
            "pending_raw_bytes": pending_bytes,
            "event_identity_bytes": identity_bytes,
            "encoding_calls": self.encoding_calls,
            "encoded_frames_including_prefix_reencodes": self.encoded_frames,
        }
