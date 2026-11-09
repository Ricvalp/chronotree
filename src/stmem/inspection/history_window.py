"""Read-only recent-execution-window control for a frozen FULL policy.

Preserve conditioning and the original already-encoded block leaves. Keep the
latest N completed execution blocks and every pending execution leaf. No raw
observation is reencoded or edited, and all original current/task/clock inputs
and flow-noise sampling remain in the ordinary OnlinePolicy path.
"""

from __future__ import annotations

import time


def select_recent_execution_records(evidence, records, completed_blocks=2):
    if isinstance(completed_blocks, bool) or not isinstance(completed_blocks, int) or completed_blocks < 0:
        raise ValueError("completed_blocks must be a nonnegative integer")
    archive = evidence.archive
    blocks = []
    for block_id, leaves in archive.completed_blocks.items():
        if not leaves:
            continue
        if len({leaf.source for leaf in leaves}) != 1:
            raise ValueError("An encoded block cannot cross source boundaries")
        if leaves[0].source == "execution":
            blocks.append((block_id, leaves))
    # Archive guarantees source-local chronological append order. Keeping its
    # original leaves also preserves complete-block contextualization exactly.
    retained = blocks[-completed_blocks:] if completed_blocks else []
    pending = list(archive.pending.get("execution", ()))
    allowed = {record.node_id for _, leaves in retained for record in leaves}
    allowed.update(record.node_id for record in pending)
    selected = [record for record in records if record.source == "conditioning" or record.node_id in allowed]
    execution = [record for record in selected if record.source == "execution"]
    bounds = None if not execution else [min(record.time_min for record in execution),
                                        max(record.time_max for record in execution)]
    anchors = sorted({float(record.time_anchor) for record in execution})
    info = {
        "requested_completed_execution_blocks": completed_blocks,
        "available_completed_execution_blocks": len(blocks),
        "retained_completed_execution_blocks": len(retained),
        "retained_completed_execution_block_ids": [str(block_id) for block_id, _ in retained],
        "pending_execution_leaves": len(pending),
        "original_history_tokens": len(records), "read_history_tokens": len(selected),
        "conditioning_tokens": sum(record.source == "conditioning" for record in selected),
        "execution_tokens": len(execution),
        "execution_recorded_time_bounds": bounds,
        "execution_time_anchor_bounds": None if not anchors else [anchors[0], anchors[-1]],
        "execution_distinct_time_anchors": len(anchors),
        "clock_units": "native_source_local_recorded_frame",
        "conditioning_preserved": True, "current_goal_clocks_preserved": True,
        "encoder_recomputed": False,
    }
    return selected, info


class RecentExecutionOnlinePolicy:
    """Temporarily filter the FULL action-head read, without editing the archive."""

    def __init__(self, online, completed_blocks=2):
        if online.method != "FULL" or online.intervention != "none":
            raise ValueError("Wrap only ordinary FULL without another intervention")
        if isinstance(completed_blocks, bool) or not isinstance(completed_blocks, int) or completed_blocks < 0:
            raise ValueError("completed_blocks must be a nonnegative integer")
        self.online = online
        self.completed_blocks = completed_blocks
        self.reads = []

    def __getattr__(self, name):
        return getattr(self.online, name)

    @property
    def update_seconds(self):
        return self.online.update_seconds

    @update_seconds.setter
    def update_seconds(self, value):
        self.online.update_seconds = value

    def reset(self, reset, seed):
        self.reads = []
        return self.online.reset(reset, seed)

    def observe(self, *args, **kwargs):
        return self.online.observe(*args, **kwargs)

    def decide(self):
        policy = self.online.policy
        original = policy.context
        had_instance_override = "context" in policy.__dict__
        instance_override = policy.__dict__.get("context")
        captured = []

        def window_context(evidence, records, **kwargs):
            started = time.perf_counter()
            selected, info = select_recent_execution_records(evidence, records, self.completed_blocks)
            info["filter_seconds"] = time.perf_counter() - started
            info["current_source_index"] = int(evidence.current.source_index)
            info["decision_index"] = int(self.online.decision_index)
            captured.append((selected, info))
            return original(evidence, selected, **kwargs)

        policy.context = window_context
        try:
            actions, timing = self.online.decide()
        finally:
            if had_instance_override:
                policy.context = instance_override
            else:
                delattr(policy, "context")
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one FULL context read per decision")
        selected, info = captured[0]
        self.reads.append(info)
        timing["original_archived_history_tokens"] = info["original_history_tokens"]
        timing["history_tokens"] = len(selected)
        timing["active_tokens"] = len(selected)
        timing["records_fetched"] = len(selected)
        timing["represented_leaf_counts"] = [record.count for record in selected]
        timing["resolution_depths"] = [record.depth for record in selected]
        timing["history_window"] = info
        timing["selected_transfer_payload_bytes"] = sum(
            (record.content_sum.numel() + 23) * 4
            for record in selected if record.content_sum.device != policy.device
        )
        return actions, timing
