"""Repository tests never publish runs using a developer's ambient W&B login."""

import pytest


@pytest.fixture(autouse=True)
def offline_tracking(monkeypatch):
    for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR", "WANDB_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
