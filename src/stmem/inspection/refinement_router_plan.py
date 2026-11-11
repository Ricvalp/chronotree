"""Prepare a bounded, disabled later-router blueprint without mislabeling data.

This is executable preparation, not a label-generation or Slurm-submit command.
The missing schema/controller integrations are explicit machine-readable blockers.
The primary three-student experiment does not depend on this conditional branch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

from stmem.config import digest, source_revision
from stmem.inspection.refinement_data import primary_students, read_json, reference_identity, verify_manifest
from stmem.inspection.refinement_utility import validate_student_endpoint
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_checkpoint


BLOCKERS = (
    "The existing router loader accepts only velocity/final_action teacher-agreement schema-v2 labels. Implement a distinct expert-FM first-move/completed-packet schema and writer with raw signed losses, route prefixes and student provenance.",
    "Add observable remaining-budget/feasible-cost feature encoding with matched scorer dimensions for both targets/readers; existing LocalRouter features alone do not include these new scalars.",
    "Connect trained scorer predictions to the shared exact-coverage IndexedArchive local/packet controller, with all proposal/index/descendant/scoring costs recorded.",
    "Existing closed-loop router validation requires the label-generating compact hash. Add an explicit diagnostic crossed-student protocol validating compatible frozen representations without weakening the production exact-student guard.",
    "One router-conditioned student adaptation cycle needs a frozen-router route provider and new student-specific labels after adaptation. Prepared adaptation configs do not implement that provider.",
)


def validate_training_pool(manifest):
    """Do not revive diagnostic or old router-validation episodes for fitting."""
    def name(row):
        return row["task"], row["episode_id"]
    pool = list(map(name, manifest["training_pool"]))
    excluded = set(map(name, manifest["diagnostic_episodes"])) | set(map(name, manifest["reserved_validation_episodes"]))
    partitions = {name(row): row["partition"] for row in manifest["episode_partitions"]}
    if not pool or len(pool) != len(set(pool)):
        raise ValueError("Router fitting requires a nonempty unique common training pool")
    if set(pool) & excluded or any(partitions.get(episode) != "train" for episode in pool):
        raise ValueError("Router fitting pool includes diagnostic, reserved validation or unknown episodes")
    tasks = set(manifest["configuration"]["data"]["tasks"])
    if {task for task, _ in pool} != tasks:
        raise ValueError("Every configured task must remain represented in the router fitting pool")
    return {"episodes": len(pool), "by_task": dict(Counter(task for task, _ in pool)),
            "episode_pool_sha256": digest(sorted(pool)), "excluded_episodes": len(excluded)}


def prepare(manifest_path, output, *, checkpoints, finding=None,
            configuration="configs/inspection/refinement_router_conditional.yaml"):
    manifest = verify_manifest(manifest_path)
    cfg = yaml.safe_load(Path(configuration).read_text())
    if cfg.get("enabled") is not False or cfg.get("maximum_students") != 2:
        raise ValueError("This conditional preparation must remain disabled and bounded to at most two students")
    # The parent study owns replanning cadence; the conditional template must
    # not silently restore the older teacher study's ten-target protocol.
    cfg["closed_loop"]["executed_targets"] = manifest.get("protocol", {}).get(
        "execute_targets", cfg["closed_loop"]["executed_targets"])
    if not 1 <= len(checkpoints) <= 2 or set(checkpoints) - set(primary_students(manifest)):
        raise ValueError("Prepare one or at most two explicitly selected primary student variants")
    pool = validate_training_pool(manifest)
    teacher_hash = manifest["dependencies"]["teacher_sha256"]
    students = {}
    for variant, path in sorted(checkpoints.items()):
        path = Path(path).resolve()
        state = load_checkpoint(path)
        if state.get("stage") != "compact":
            raise ValueError("Router labels require a frozen compact student")
        if state.get("provenance", {}).get("dependencies", {}).get("teacher") != teacher_hash:
            raise ValueError("Selected student depends on a different teacher")
        if variant != "S0":
            training_identity = validate_student_endpoint(manifest, manifest_path, variant, state, "train")
            if state.get("step") != training_identity["endpoint"]:
                raise ValueError("Primary student selection must use its predeclared primary endpoint")
        else:
            reference_hash, reference_step = reference_identity(manifest)
            if state.get("step") != reference_step or file_sha256(path) != reference_hash:
                raise ValueError("S0 must be the unchanged 90k reference")
            training_identity = {"initialization": "production_reference", "endpoint": reference_step}
        students[variant] = {"path": str(path), "sha256": file_sha256(path), "step": state["step"],
                             "teacher_sha256": teacher_hash, **training_identity}
    matched = {(row["initialization"], row["endpoint"], row["step"]) for variant, row in students.items()
               if variant != "S0"}
    if len(matched) > 1:
        raise ValueError("Conditional router students must share initialization and declared training duration")
    development_finding = None
    if finding is not None:
        development_finding = read_json(finding)
        if development_finding.get("partition") != "train" or development_finding.get("experiment_id") != manifest["experiment_id"]:
            raise ValueError("Activation finding must be development-only evidence from this experiment")
        development_finding = {"path": str(Path(finding).resolve()), "sha256": file_sha256(finding),
                               "record": development_finding}
    nstudents, nreaders, ntargets = len(students), len(cfg["readers"]), len(cfg["credit_targets"])
    contexts, candidates = cfg["fitting_states_per_student"], cfg["candidate_limit"]
    draws = cfg["discovery_draws"] + cfg["confirmation_draws"]
    crossed = [{"label_student": label_student, "evaluation_student": eval_student,
                "reader": reader, "credit_target": target, "budget": cfg["target_budget"],
                "status": "prepared_not_activated"}
               for label_student in students for eval_student in students
               for reader in cfg["readers"] for target in cfg["credit_targets"]]
    body = {"schema": cfg["schema"], "status": "prepared_conditional_not_executable_training",
            "enabled": False, "source": source_revision(), "experiment_id": manifest["experiment_id"],
            "parent_manifest": str(Path(manifest_path).resolve()), "configuration": cfg,
            "students": students, "training_pool": manifest["training_pool"], "pool_audit": pool,
            "development_finding": development_finding, "activation_blockers": list(BLOCKERS),
            "fitting_plan": {"states_per_student": contexts, "shared_state_manifest_required": True,
                             "readers": cfg["readers"], "credit_targets": cfg["credit_targets"],
                             "fit_jobs": nstudents * nreaders * ntargets,
                             "total_fit_optimizer_updates": nstudents * nreaders * ntargets * cfg["fit"]["steps"],
                             "compact_velocity_example_upper_bound": nstudents * contexts * nreaders * (1 + 2 * candidates) * draws,
                             "teacher_velocity_examples_shared": contexts * draws,
                             "excludes_optional_refresh_cost": True,
                             "runtime_estimate": "unmeasured; benchmark new label writer before allocation"},
            "crossed_offline_matrix": crossed,
            "closed_loop_protocol": cfg["closed_loop"],
            "scientific_note": "Prepared follow-up only. Existing teacher-agreement labels are not expert/packet labels and are never renamed or reused here."}
    body["plan_id"] = digest(body)
    output = Path(output)
    if output.exists() and read_json(output) != body:
        raise FileExistsError("Conditional router plan changed; choose a new output path")
    atomic_json(body, output)
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, metavar="STUDENT=PATH")
    parser.add_argument("--development-finding", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/inspection/refinement_router_conditional.yaml"))
    args = parser.parse_args(argv)
    checkpoints = {}
    for value in args.checkpoints:
        if "=" not in value:
            parser.error("Use STUDENT=PATH")
        name, path = value.split("=", 1)
        if name in checkpoints or not path:
            parser.error("Checkpoint names must be distinct and paths nonempty")
        checkpoints[name] = path
    result = prepare(args.manifest, args.output, checkpoints=checkpoints,
                     finding=args.development_finding, configuration=args.config)
    print(json.dumps({"plan": str(args.output), "status": result["status"],
                      "fitting_plan": result["fitting_plan"], "activation_blockers": result["activation_blockers"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
