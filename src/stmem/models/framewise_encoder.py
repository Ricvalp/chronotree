"""Matched leaf-count attribution variant, restricted to each center's own frame."""

from .spacetime_encoder import SpacetimeEncoder


class FramewiseEncoder(SpacetimeEncoder):
    def __init__(self, *args, **kwargs):
        kwargs["framewise"] = True
        super().__init__(*args, **kwargs)
