"""Keep random compact checkpoint selection distinct from learned routing."""


def validate_selection_protocol(method, selection_family, router_checkpoint, *, checkpoint_stage, split):
    if selection_family is not None:
        if selection_family not in ("tree", "flat") or method != selection_family.upper():
            raise ValueError("--selection-family must match --method TREE or FLAT")
        if router_checkpoint is not None:
            raise ValueError("Random --selection-family cannot use --router-checkpoint")
        if checkpoint_stage != "compact":
            raise ValueError("Random --selection-family requires a compact-stage checkpoint")
        if split != "val":
            raise ValueError("Random --selection-family is validation-only")
    elif method in ("TREE", "FLAT") and router_checkpoint is None:
        raise ValueError("TREE/FLAT requires --router-checkpoint or --selection-family")
