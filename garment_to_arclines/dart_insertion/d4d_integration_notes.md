# D4D integration notes

Working notes from understanding D4D's rewrite/SRD architecture, in preparation
for eventually registering `insert_dart` (transform.py) as a D4D-compatible
rewrite. Written up so we can pick back up without re-deriving everything.

## Q1 — How is a rewrite defined?

A rewrite is a small piece of *data*, not a class with behavior. Concretely:
`ShapeRewriteSpec(type: ShapeRewriteType, args: tuple[float, ...])`
(`d4descent/src/d4descent/objects/arclines.py:264`). `type` says which kind of
edit (Split, Merge, ToLine, ToArc, AddHole, RemoveHole, ...); `args` is almost
always index numbers into `shape.primitives` (which piece(s) it touches), not
geometric values — except `AddHole`, whose args are real numbers (x, y,
radius, reversed, n_segments) since it isn't attached to anything existing.

One function, `Shape.apply_rewrite(spec)` (`arclines.py:727`), is a single
dispatch (`if spec.type == ...`) that knows how to perform every kind. There's
no per-type class hierarchy. `Split` always cuts at a fixed midpoint (t=0.5)
— the spec doesn't carry a learnable "where."

Locations are always "the Nth thing in the primitive list" — no semantic
edge/vertex naming like our panel's edge labels.

## Q2 — How SRD samples and applies rewrites (the discrete step)

Most optimizer steps are just ordinary gradient descent. Every `propose_every`
steps (default 50, `optimizer.py`), a rewrite round fires:

1. **Enumerate** — `Shape.generate_rewrite_specs()` (`arclines.py:833`) walks
   every current primitive and proposes one candidate spec per applicable
   rewrite type (every line could split/become an arc, every arc could
   split/become a line, adjacent pairs sharing an endpoint could merge).
   Add-hole candidates are generated separately, scattered randomly or on a
   grid — not tied to existing geometry.
2. **Apply tentatively** — every candidate spec is actually applied right
   away via `apply_rewrite`, producing a full batch of "what if" shapes
   (`tasks/arclines.py:135`).
3. **Score** — each candidate gets a couple of its own gradient steps
   (`proposal_steps`, default 2) to settle, then loss + a simplicity penalty
   (more pieces costs more) is measured.
4. **Combine** — candidates that improved meaningfully are sorted by
   improvement and accepted greedily, skipping any that conflict (share a
   touched primitive index) with an already-accepted one
   (`tasks/arclines.py:141-194`). All accepted, non-conflicting edits are
   applied together to produce the new base shape.
5. If nothing helped, nothing changes; normal gradient descent resumes.

## Q3 — How does a rewrite's continuous parameter reach the gradient step?

It doesn't, via the spec. Every point coordinate and every arc's curvature
`k` is *already* a raw, always-tracked number — `Line`/`Arc` dataclasses hold
plain tensors (`arclines.py:58-92`), and across the whole population these
get flattened into two big tensors, `control_points` and `ks`
(`ShapeCollection.parameters()`, `arclines.py:1793-1794`), which is what the
optimizer actually backprops through every normal step.

A rewrite's job is purely *structural* — deciding what points/curvatures now
exist — not assigning them a value. E.g. `ToArc` creates the new arc with
`k=0` (straight, no visual jump) and leaves it for gradient descent to move
away from zero over subsequent steps (`line_to_arc2`, `arclines.py:402-403`).

**Key tension for us:** our dart's `t` is one clean semantic scalar
interpreted by a whole transform function. D4D has no such concept — its
"parameters" are always raw, meaning-free coordinates/curvatures that get
tugged independently. This is the crux of the open design question below.

## Q4 — How does placement ("where to fire") work?

No smart targeting — brute-force enumeration every rewrite round. Split/
merge/line↔arc are proposed for *every* current primitive (or every
shared-endpoint adjacent pair, for merge), by looping over `self.primitives`.
Add-hole is the other pattern: untethered from existing geometry, scattered
across the canvas at random/grid positions.

So there are two existing placement patterns to choose from for a new
rewrite: **edge-attached** (index-based, like split/merge) or **free-floating**
(like add-hole).

## Resolved: no placement search needed

Earlier drafts of this plan assumed `AddDart` needed to search over apex
position and side-seam position, mirroring how `AddHole` scatters candidate
centers. That was wrong — corrected in conversation:

- **Apex is anatomical, not a design choice.** It's the bust point of a
  given body; it doesn't move regardless of which dart pattern removes
  fabric there. It should be a fixed, known input (same as every prior
  phase), never something to search over.
- **`side_seam_t` is also fixed for now.** The transform's math (hinge on
  the armhole chain, the translation formula, etc.) is derived specifically
  for a side-seam dart — it hasn't been checked against other entry edges
  (shoulder, armhole, waist darts), so there's currently only one dart
  *type* to fire, not several placements to choose between.
- **What a real optimizer actually needs to decide is just two things**:
  *whether* to fire this dart at all (binary — SRD's existing generic
  accept-if-better logic already handles this for any rewrite type, nothing
  dart-specific to build), and *how much spread* (`t`) to use if it fires.
- **Future direction, not now**: once other dart-type transforms exist
  (shoulder dart, French dart, etc. — each independently validated the way
  this one was through Phases A-E), *those* become the real candidate set —
  SRD choosing between different rewrite *types* firing at their own fixed,
  known positions, the same way it already chooses between Split/Merge/
  ToArc/ToLine today. Not a position search within one dart type.

## Phase F: insert_dart wrapped as a rewrite (done)

Built `rewrite.py`: `AddDartSpec(apex_raw, side_seam_t, t)` (mirrors
`ShapeRewriteSpec`'s shape) and `apply_add_dart(panel, spec) -> DartResult`
(mirrors `apply_rewrite(spec)`), wrapping `insert_dart` directly with no new
geometry logic. `apex_raw`/`side_seam_t` are fixed, known inputs; `t` is
forced to one constant value (`inches_to_t(panel, 0.5)`) — firing is
unconditional, there's no accept/reject step yet since there's still no
objective function. Tested (`test_rewrite.py`, 6 tests, curved-synthetic +
real panel) and visually confirmed (`visualize_phase_f.py` ->
`out/phase_f/`) that the wrapper reproduces `insert_dart` exactly.

Deliberately NOT touching `d4descent`'s `ShapeRewriteType`/`apply_rewrite` —
this stays a standalone module, same convention as the `split_line`
workaround in `chains.py`.

Also added `generate_add_dart_specs(panel, apex_raw, side_seam_t, t_inches)`
— the "generate_rewrite_specs" side of the pair, but not a search: since the
transform depends on a full set of labeled edge roles all present at once
on one panel, there's at most one valid candidate per panel, not several to
choose between. Mirrors `RemoveHole`'s pattern (propose nothing if the
shape doesn't qualify) rather than `Split`/`AddHole`'s enumeration/sampling
patterns — checks `panel.edge_labels` covers `{center_front, collar,
shoulder, armhole, side_seam, hem}`, returns `[]` otherwise. 10 tests total
now (added eligible/ineligible-panel cases for the generator).

## Checked against the paper's 4 design guidelines (D4D.pdf, Table 1)

- **Reversibility** (if A→B exists, B→A should too): we already have
  `undo_dart()` (Phase C), so a `RemoveDartSpec`/`apply_remove_dart`
  companion is conceptually cheap — but there's a real snag: `undo_dart`
  takes a `DartResult` (hinge, theta, trans, ...), not a plain `Shape`, so it
  can only reverse a dart *we just inserted in the same call*, not an
  arbitrary already-darted shape SRD might encounter later (e.g. after other
  rewrites ran in between). A true reverse rewrite needs that state stashed
  somewhere recoverable (D4D's `ShapePayload`?) or re-derived from the
  boundary geometry — deliberately not built yet, flagged for a deliberate
  follow-up rather than building a version that only round-trips its own
  output.
- **Jump Continuity** (negligible instantaneous change on fire): already
  satisfied, and was designed in back in Phase C for unrelated reasons —
  firing at `t=0` is an exact geometric match to the un-darted panel (the
  whole reason for the `min(target_setback, gap)` construction in
  `transform.py`'s docstring).
- **Local Geometric Control** (a rewrite should exist that changes the shape
  near one point without disturbing distant parts — paper's own example is
  literally "Add-Anywhere"): **honest mismatch.** Unlike D4D's existing
  AL-grammar rewrites (Split/Merge/ToArc/ToLine/AddHole), which only ever
  touch 1-2 primitives and leave everything else byte-identical, `AddDart`
  rebuilds almost the entire boundary (only `collar_chain` and
  `armhole_cf_side` survive untouched) — rotating the whole upper region and
  translating the whole lower region is inherent to how a dart physically
  closes fabric, not an artifact of implementation choices. The paper is
  explicit that grammars need not satisfy every guideline for every rewrite,
  so this isn't disqualifying, but it's a real difference in kind from the
  rest of the grammar, worth remembering if optimization behaves oddly later.
- **Repairability** (a rewrite should exist to project an invalid state back
  to valid, if constraints exist): not relevant yet since `t` is manually
  kept under the domain ceiling (`t_max = a*(1+sin(alpha))`) rather than
  optimized. Becomes relevant if `t` ever goes live (option (b) below) —
  gradient steps could push it past `t_max` and hit the `arcsin` domain
  error, which is exactly what a repair/clamp step (mirroring D4D's
  `project_to_valid_()`) would exist to prevent. Not built.

## Reversibility: built (with a documented limitation)

Added `RemoveDartSpec` (empty -- nothing left to choose, undo is fully
determined by the `DartResult` it's given) and `apply_remove_dart(result,
spec) -> dict` wrapping `undo_dart` directly, in `rewrite.py`. Tested
(`test_apply_remove_dart_reverses_apply_add_dart`, 2 cases) and visually
confirmed (`visualize_phase_f.py`, 3-panel: unfired -> fired -> reversed) --
the reconstructed points land exactly back on the original outline.

Limitation stays as documented in `RemoveDartSpec`'s docstring: this only
reverses a dart *just fired in the same call*, since `undo_dart` needs the
`DartResult`'s internal state (hinge/theta/trans), not just a plain `Shape`.
Reversing an arbitrary already-darted shape found later (e.g. after other
rewrites ran) would need that state stashed somewhere recoverable or
re-derived from geometry -- not built, deliberately deferred.

**Upgraded further**: `apply_remove_dart` now returns `UndoDartResult`
(`boundary`, `reconstructed`), a real re-assembled closed boundary built
from `undo_dart`'s points -- not just a bag of loose points -- so it's an
actual shape to draw and compare, matching `DartResult`'s contract.
Visualized as a 3rd panel in `visualize_phase_f.py`: the reconstructed
boundary drawn as its own polygon, sitting exactly on top of the original
outline.

**A second, more specific limitation surfaced while building this**: full
*topological* reversibility isn't achieved, only *geometric* reversibility.
The hinge split introduced while inserting the dart
(`split_chain_by_arc_length`) is permanent -- `undo_dart` reverses the rigid
motion, not the split itself -- so the reconstructed armhole legitimately
has one more primitive than the original (M+1 vs M forever after). A true
`Split`/`Merge`-style exact restoration would need `apply_remove_dart` to
also merge the hinge-adjacent primitives back together. Not built; verified
this is what's happening (not a bug) via endpoint + arc-length-match checks
in `test_apply_remove_dart_boundary_matches_original_panel` instead of
exact primitive-list equality.

## Mechanism note: where new boundary vertices come from

`insert_dart` doesn't call D4D's `split_line`/vertex-adding primitives —
it builds the whole new primitive list directly via geometry math. But
conceptually, the new points it introduces map cleanly onto D4D's own
vocabulary: the dart point (both copies) is a **split** of the side-seam
edge at a fraction along it (same idea as D4D's `Split`, just computed with
`fraction_along` instead of calling `split_line`); the tip and new corner
are more like **add-anywhere** points, not derived from splitting an
existing edge. This framing is useful for explaining the design but isn't
needed as an actual implementation mechanism right now, since apex/
side_seam_t are fixed inputs, not searched — routing through D4D's literal
split/add-vertex code would only matter if those become searchable later.

**The real fork — what happens to `t` after the dart fires:**

- **(a) Fixed once triggered.** The transform picks a reasonable `t` from
  the geometry at fire time (or it's baked into the spec) and that's final.
  SRD's role is then *purely* "where/whether to fire" — same as every other
  rewrite — and never touches openness again. Smaller lift, fits D4D's
  existing machinery with no new plumbing.
- **(b) Continuously refined afterward.** `t` lives on as its own tracked,
  differentiable scalar (separate from the normal `control_points`/`ks`
  pool), and the dart's boundary is recomputed from `t` fresh every forward
  pass — same relationship `k` has to arcs, but for a whole semantic
  transform instead of one raw number. Preserves geometric validity by
  construction as `t` moves, but needs new plumbing D4D doesn't have today
  (it doesn't currently track/backprop through anything but raw points and
  `k`).

Leaning toward prototyping (a) first since it's a much smaller lift and
reuses existing machinery — worth checking empirically whether loss +
simplicity alone keep a fixed-`t` dart looking reasonable before building (b).
Not yet decided — this is where we paused, mid-discussion of (a) vs (b),
to consolidate notes before continuing.

## Status / where we paused

All four questions answered. Phase F (wrapping `insert_dart` as a rewrite,
forced fire, fixed `t`) is implemented, tested, and visually confirmed. The
open thread is still fixed-`t` (a) vs. live-`t` (b) from Q3 above — i.e.
whether `t` gets fixed once a dart fires or stays a continuously-refined
parameter — which matters once there's an actual objective function and
accept/reject step to build. Not yet decided; nothing beyond Phase F has
been implemented.
