"""Deploy the expert-utility packet scorer on the live causal archive.

This experimental controller scans indexed topology and re-scores after each
packet. It is not the original once-per-node heap and makes no O(B) search claim.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from stmem.data.hierarchy_cache import PAYLOAD_FIELDS, SCALAR_FIELDS, STRING_FIELDS
from stmem.inspection.refinement_candidates import IndexedArchive
from stmem.memory.flat import FlatState
from stmem.memory.frontier import Selection
from stmem.memory.hierarchy import walk_records
from stmem.memory.records import record_bytes
from stmem.memory.utility_routing import candidate_features, query_features, route
from stmem.training.packed_context import _empty_columns, _take, columns_bank, flat_residual_columns


def evaluation_settings(state, *, controller=None, selection=None, budget=None):
    """Bind controller/control interventions to an exact, trained utility schema."""
    if state.get("schema") != "utility_router_checkpoint_v1":
        if controller is not None or selection is not None:
            raise ValueError("Router controller/selection overrides require an expert-utility router checkpoint")
        return None
    protocol = state["utility_protocol"]
    if budget is not None and budget != protocol["budget"]:
        raise ValueError("Expert-utility router budget differs from its fitting protocol")
    controller, selection = controller or "packet", selection or "learned"
    if controller not in ("packet", "local") or selection not in ("learned", "random", "uniform"):
        raise ValueError("Invalid expert-utility controller/selection")
    return {"protocol": protocol, "controller": controller, "selection": selection,
            "scope": "development_only", "comparison": "credit_target" if controller == "packet" else "controller"}


def record_columns(records, width):
    """Bulk CPU transfer, identical field/reduction convention to cached labels."""
    if not records:
        return _empty_columns(width)
    result = {key: torch.stack([getattr(record, key).detach() for record in records]).float().cpu()
              for key in PAYLOAD_FIELDS}
    result.update({key: np.asarray([getattr(record, key) for record in records]) for key in SCALAR_FIELDS})
    result.update({key: np.asarray([getattr(record, key) for record in records], dtype=object) for key in STRING_FIELDS})
    return result


class LiveIndexedArchive:
    def __init__(self, archive, width, clocks):
        self.roots = archive.roots
        self.records = walk_records(self.roots)
        positions = {record.node_id: i for i, record in enumerate(self.records)}
        children = np.asarray([[positions[child.node_id] for child in record.children]
                               if record.children else [-1, -1] for record in self.records], dtype=np.int64).reshape(-1, 2)
        self.columns = record_columns(self.records, width)
        index = {key: self.columns[key] for key in (*SCALAR_FIELDS, *STRING_FIELDS)}
        index.update(children=children, anchor=self.columns["anchor"].numpy())
        # Preserve the same chronological leaf reduction order as cached blocks.
        self.leaves = list(archive.leaves)
        rows = np.asarray([positions[record.node_id] for record in self.leaves], dtype=np.int64)
        self.archive = IndexedArchive(index, [positions[record.node_id] for record in self.roots], rows)
        self.leaf_columns = _take(self.columns, rows)
        self.width, self.clocks = width, clocks
        self.index_bytes = sum(value.nbytes for value in index.values() if isinstance(value, np.ndarray))
        self.payload_bytes = sum(value.numel() * value.element_size() for value in self.columns.values()
                                 if isinstance(value, torch.Tensor))
        if len(rows):
            execution = self.leaf_columns["source"] == "execution"
            if execution.any() and np.max(self.leaf_columns["time_max"][execution]) >= float(clocks[1]):
                raise ValueError("Current/future execution evidence entered the router archive")

    def selected_columns(self, context):
        if context["reader"] == "TREE":
            return _take(self.columns, np.asarray(context["rows"], dtype=np.int64))
        return flat_residual_columns(self.leaf_columns, *self.archive.flat_arrays(context))

    def bank(self, context):
        return columns_bank(self.selected_columns(context), self.clocks, self.width)

    def selected_records(self, context):
        context = self.archive.validate(context)
        if context["reader"] == "TREE":
            records = [self.records[row] for row in context["rows"]]
        else:
            records = FlatState(tuple(self.leaves), {self.records[row].node_id for row in context["rows"]}).records
        if len(records) != context["token_count"]:
            raise AssertionError("Online residual accounting differs from indexed router")
        return records


class UtilityOnlineRouter:
    def __init__(self, router, *, reader, budget, protocol, controller="packet", selection="learned"):
        if controller not in ("packet", "local") or selection not in ("learned", "random", "uniform"):
            raise ValueError("Invalid expert-utility controller or control selection")
        if router.training or any(parameter.requires_grad for parameter in router.parameters()):
            raise ValueError("Online utility router must be frozen")
        if budget != protocol["budget"]:
            raise ValueError("Utility router evaluation must use its trained budget")
        self.router, self.reader, self.budget = router, reader, budget
        self.protocol, self.controller, self.selection = protocol, controller, selection
        self.width = router.width
        self.transferred_bytes = 0

    def reset(self):
        self.transferred_bytes = 0

    @torch.no_grad()
    def select(self, evidence, *, seed):
        started = time.perf_counter()
        clocks = torch.tensor([evidence.clocks.get("conditioning", 0.), evidence.clocks.get("execution", 0.)])
        live = LiveIndexedArchive(evidence.archive, self.width, clocks)
        current = columns_bank(record_columns(evidence.current_records, self.width), clocks, self.width)
        text = evidence.text.detach().float().cpu()
        # record_columns promotes payloads to float32 before its CPU transfer.
        input_transfer = sum(getattr(record, key).numel() * 4
                             for record in (*live.records, *evidence.current_records)
                             for key in PAYLOAD_FIELDS if getattr(record, key).device.type != "cpu")
        if evidence.text.device.type != "cpu":
            input_transfer += evidence.text.numel() * 4
        index_seconds = time.perf_counter() - started
        device = next(self.router.parameters()).device
        rng = random.Random(seed)
        transferred = 0
        hydrated = 0

        def score(context, packets):
            nonlocal transferred, hydrated
            if self.selection == "random":
                return [rng.random() for _ in packets]
            if self.selection == "uniform":
                return [float(packet["support_count"]) for packet in packets]
            cost = packets[0]["cost"]
            query = query_features(current, text, live.bank(context), clocks,
                                   width=self.width, budget=self.budget, packet_cost=cost)
            columns = _take(live.columns, np.asarray([packet["region"] for packet in packets], dtype=np.int64))
            candidates = candidate_features(columns, width=self.width, cost=cost, budget=self.budget)
            hydrated += context["token_count"] + len(packets)
            if device.type != "cpu":
                transferred += 4 * (query.numel() + candidates.numel() + len(packets))
            return self.router(query.to(device), candidates.to(device)).float().cpu().numpy()

        context, work = route(live.archive, self.reader, self.budget, score, seed=seed,
                              candidate_limit=self.protocol["candidate_limit"],
                              packet_max_cost=self.protocol["packet_max_cost"], mode=self.controller)
        records = live.selected_records(context)
        self.transferred_bytes += input_transfer + transferred
        stats = {f"utility_{key}": value for key, value in work.items() if key != "steps"}
        stats.update(utility_index_seconds=index_seconds, utility_index_bytes=live.index_bytes,
                     utility_index_payload_bytes=live.payload_bytes, utility_scoring_transfer_bytes=transferred,
                     utility_index_record_count=len(live.records),
                     utility_input_transfer_bytes=input_transfer,
                     utility_routing_transferred_bytes=input_transfer + transferred,
                     utility_routing_transferred_bytes_cumulative=self.transferred_bytes,
                     utility_feature_records_fetched=hydrated, utility_selection=self.selection,
                     utility_route_decisions=len(work["steps"]),
                     utility_total_selection_seconds=time.perf_counter() - started,
                     nodes_scored=work["candidates_scored"] if self.selection == "learned" else 0,
                     active_tokens=len(records), records_fetched=len(records),
                     selected_payload_bytes=record_bytes(records), selected_leaves=sum(record.count == 1 for record in records))
        return Selection(records, stats)
