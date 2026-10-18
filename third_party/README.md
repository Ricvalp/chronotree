The public sources used for protocol audit are pinned in [sources.lock.json](sources.lock.json). Downloaded checkouts live in ignored `third_party/checkouts/`; no nested repository is committed.

```bash
python3 scripts/fetch_sources.py
bash scripts/setup_sim.sh
```

| Source | Revision | License | Use |
|---|---|---|---|
| [RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark) | `6503b26e00ff338611fceb039417fbc64826ccc8` | Apache 2.0 | Official environment; source audit |
| [ManiSkill fork](https://github.com/YinpeiDai/ManiSkill) | `07be6fbc66350ddca200abfb0a11b692f078f7fd` | Apache 2.0; assets have additional notices | Simulator dependency |
| [Official policy suite](https://github.com/RoboMME/robomme_policy_learning) | `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` | Apache 2.0; optional Gemma terms separate | Alignment audit only; no policy weights or code copied |
| [Training HDF5](https://huggingface.co/datasets/Yinpei/robomme_data_h5) | `a5e4e25ffe8af34f64944f9533d06455ce5f8337` | Dataset card declares Apache 2.0 | Released demonstrations |

License copies are in `licenses/`. ManiSkill's asset notices remain in its installed package and checkout; see its `LICENSE-3RD-PARTY`. No upstream patches were required. The official MME-VLA implementation is JAX/OpenPI. This repository implements a separate compact PyTorch policy and does not inherit any foundation-model weights.

The source lock also records the measured task-archive sizes and Hugging Face LFS SHA256 values. The downloader resolves the pinned revision, prints compressed sizes before transfer, verifies the archive hash, refuses links/traversal on extraction and records actual extracted size. `--episodes` limits preparation only: public storage is one whole archive per task.

Cite: Dai et al., *RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies*, arXiv:2603.04639, 2026.
