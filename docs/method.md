# Query-adaptive space-time evidence

The implementation tests whether a fixed historical token budget can retain coarse
coverage of all available observations while exposing selected evidence at finer
resolution. TREE and FLAT consume the same frozen fine leaves and shared compact
policy. A growing CPU archive is permitted: bounded active context does not mean
bounded total storage.

## Causal observations and local evidence

`Frame` is an allowlisted point cloud, observed proprioception, preceding command
with an availability bit, source-local recorded-frame timestamp, source, and event
identity. Neither episode duration nor expert future actions enter the encoder or
router. Conditioning and execution have separate clocks and banks. The current
execution observation is represented separately and excluded from history.

The joint encoder allocates center candidates to every observed frame. With the
core configuration, each eight-frame block has four candidates per frame, for 32
visual leaves. An incomplete prefix uses only its observed frames and their
corresponding quotas. Points are sorted lexicographically by observed xyz/RGB and
modest farthest-point sampling selects centers within each temporal stratum. This
makes center selection invariant to point order, including deterministic tie
handling. A stratum with fewer valid points has fewer occupied leaves; missing
evidence is never fabricated to fill a tensor shape.

For each center, the code computes FP32 space/time distances within the local block
only. Defaults are spatial radius 0.3 metres, temporal radius 8 recorded frames,
at most 128 neighbors, and center chunks of 16. A point MLP reads xyz/RGB; a
nonlinear edge MLP also reads relative xyz and time. Normalized Gaussian weights
pool the edge features. The positive softplus denominators begin at 0.01 m² and
4 recorded-frame² and have a 1e-6 floor. These are denominators with squared
units, distinct from the hard locality radii. Available frames need not represent
contiguous physical time; no undocumented seconds conversion is made.

All-invalid neighborhoods have exactly zero content and `valid=False`; they do
not enter the archive. Each valid visual leaf has mass one, irrespective of its
Gaussian occupancy. Support bounds and observed-time intervals come from actual
neighbors. A separate robot-event MLP encodes state and the preceding observed
command once per historical frame, with an availability bit. Unavailable commands
are zeroed using a boolean selection before the network. Conditioning commands
are always unavailable. Robot-event leaves also have unit mass.

The current and historical paths share point/edge machinery. Current input uses
its configured higher center count on one frame. No global register bottleneck
is applied before archival. The optional framewise variant retains the same
center quotas and interface but restricts every visual center to its own frame;
it is prepared for later tokenizer attribution, not another default experiment.

`checkpoint_blocks=True` checkpoints each local visual block with nonreentrant
PyTorch activation recomputation during teacher training. Archive leaves remain
differentiable in offline full-prefix construction. Online block content is
encoded under `no_grad`, detached, and moved to the configured archive device
(CPU by default).

## Recoverable hierarchy and prefixes

Within each source/modality, completed blocks form balanced local binary trees.
Visual splits follow the longest anchor extent; event splits follow time. Equal
scale neighboring block roots merge as a binary counter. Parent records preserve
both children, and every leaf remains addressable. Parents contain FP32 content
sums, integer descendant counts, content maxima, count-weighted anchor means,
unions of spatial/time bounds, and structural depth/scale. Mean content is
`sum/count` before any nonlinear read adapter.

The tree is a temporal hierarchy over blocks with spatial refinement inside
visual blocks. It does not implement object tracking or arbitrary independently
chosen space/time partitions. Recoverability refers to archived leaves, not raw
point reconstruction.

An execution prefix is re-encoded only when its observed frames change and a read
requires it. Its temporary roots count toward the same token budget. Completion
replaces those temporary leaves and trees with the full observed-block version.
Completed conditioning partial blocks are finalized immediately because that
video is already available. Blocks never cross sources.

`build_archive` and `StreamingMemory` share block identities and prefix conventions.
Streaming promotes the previous current frame into history only when a new frame
arrives. Duplicate `(source,event_id)` calls are idempotent. Reset clears all
execution state and installs the new conditioning bank.

## Matched bounded reads

TREE starts from every nonempty forest and temporary-prefix root. Each move
replaces a parent by its two children. A fixed decision scorer is called once for
each encountered eligible node; a CPU heap stores its score for that decision.
Expansion continues until the budget is full or no splittable nodes remain,
including when all gains are negative. UNIFORM uses descending descendant count
with deterministic ID ties. Random valid frontiers use the same split operation.

Every frontier must cover exactly the available leaf set, without overlap.
Insufficient root budgets raise an error. `maximum_root_count` enumerates the
worst possible prefix up to declared conditioning/execution maxima; checking only
the final episode length would miss larger binary-counter forests earlier in the
episode. No roots are dropped to accommodate a small debug configuration.

FLAT scores all available fine leaves once in one batched scorer call. It ranks
the leaves and selects as many as the exact token accounting allows. Each
nonempty source/modality remainder receives one residual containing only
unselected evidence. Residuals have honest bounds, means, counts, and a
`disconnected` flag; there is no method-ID embedding. Selecting the last leaf in
a remainder replaces its singleton residual at zero cost. This replacement is
performed even when the token budget is full, and the reader fills all available
capacity. All-leaf contexts contain no empty residuals.

`FlatState` exposes the same accounting for compact training and action-utility
labels. Zero-cost moves are normalized separately; action-derived gains must
never divide by zero. These scientific invariants are tested directly, including
small-group cases where reserving a fixed residual count would underfill memory.

## Accounting and interpretation

The selected records share one read-time metadata schema. Relative age is
computed at the decision and is not frozen into cached content. Structural sums
and counts remain FP32 even when policy arithmetic uses autocast. The optional
log-count attention bias must be applied identically to all grouped readers.
Correct aggregation does not imply that a nonlinear policy on pooled tokens is
identical to the full-leaf policy.

With R roots and S binary splits, the frontier has R+S tokens. Candidate score
visits are bounded by R+2S; Python heap bookkeeping is O(B log B). These are
accounting bounds, not a measured latency advantage. Flat selection scans all
fine leaves, while tree construction, pending-prefix recomputation, archive
storage, key/index storage, record transfer, and policy decoding all remain real
costs. Storage helpers distinguish record payload from pending raw observations
and event identities; process/accelerator measurements are needed in addition
to payload counts because Python objects and allocator overhead consume memory.

The Gaussian pooling mechanism was inspected in the author's ICIL reference
(`SpacetimeEventEncoder`, revision
`3af009b458c74aa66e695e35b02f9f8801f52138`). This repository independently implements
the local mechanism in Torch; it does not import ICIL, its global register
bottleneck, its mask/privileged channels, or its TTT machinery.
