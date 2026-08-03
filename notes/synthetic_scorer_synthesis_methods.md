# Synthetic scorer synthesis methods and families

Scope: what `codes/lib/synthetic_scorers.py` (plus its alpha-grid consumer
`codes/lib/synthetic_scorer_alpha_grid.py`) actually builds, as a reference
for anyone reading the orientation/alpha-grid analyses.

## The framework

For one real donor scorer's segment-level scores `y(k,i)`, every method
builds a `dial`-indexed sweep from `dial=0` (the real donor, unchanged) to
`dial=1` (some aspect-determined endpoint). `dial=0` is shared across every
method and every family by construction, so overlaying multiple
methods/families for the same donor traces multiple paths out of one common
point.

## The three synthesis methods (`SYNTHESIS_GENERATORS`)

| name | formula | dial=1 endpoint |
|---|---|---|
| `'offset'` (default, `donor_family`) | `y_dial(k,i) = y(k,i) - dial * ebar_k` | donor with each system's constant offset from its own aspect-conditional mean removed (goes through section 3's `m`/`e`/`ebar_k` decomposition — group-mean lookup table, per-segment residual, per-system mean residual) |
| `'additive'` (`donor_family_additive`) | `y_dial(k,i) = y(k,i) + dial * aspect(k,i)` | donor plus the aspect's raw per-segment value, injected directly with no shape-preservation machinery |
| `'additive_mean'` (`donor_family_additive_mean`) | `y_dial(k,i) = y(k,i) + dial * abar_k` | donor plus the aspect's per-system mean (a per-system constant, not per-segment) |

`'offset'` is the original generator (section 4 of
`material/synthetic_scorer_construction.md`); `'additive'`/`'additive_mean'`
are siblings sharing the same dial=0/family-per-aspect shape but a simpler,
non-shape-preserving endpoint.

## Dial grids

- `DIAL_GRID`: linear, 0.0–1.0 in steps of 0.1 (11 points) — default preset
  for `'offset'`.
- `GEOM_DIAL_GRID`: geometric, `{2^-i : i=0..10}` — default preset for
  `'additive'`/`'additive_mean'`, since both saturate almost immediately on
  a linear grid (most of their movement happens between dial=0 and
  dial=0.1), so the geometric grid is the one that actually resolves their
  early sweep.
- `default_dial_preset(synthesis)` picks the right default; an explicit
  `--dial-preset` always overrides it.

## The families: which aspect each method dials on

Every donor expands into families keyed by which aspect the dial targets.
Two are common to all three synthesis methods; the other two are each
restricted to one method:

- **A (Adequacy)** — dialed on Adequacy MQM. Available for all three
  synthesis methods.
- **B (Fluency)** — dialed on Fluency MQM. Available for all three
  synthesis methods.
- **T (AllMQM)** — dialed on All MQM (`t = a + b`, the official All-MQM
  total via `mwb.mqm_scoring`'s `t`/`all_mqm` column, not a derived
  `a_pos + b_pos` sum). Built in `donor_alpha_dial_grid`
  (`codes/lib/synthetic_scorer_alpha_grid.py`) and consumed by
  `allmqm_orientation` (`codes/lib/synthetic_scorer_orientation.py`) and
  `plot_allmqm_orientation_vs_ess_pooled.py`. **T is restricted to the
  additive methods (`'additive'`, `'additive_mean'`)** — it is
  orientation-neutral by construction (`t = a + b` favors neither aspect),
  which is the property the orientation analysis relies on to treat T as a
  direction-unambiguous probe (a well-behaved metametric should prefer the
  more-T-aligned dial at every alpha, not just cross over near `alpha_0(D)`
  the way A/B do). That neutrality argument depends on the additive
  endpoint (`y + dial * aspect(k,i)` or `y + dial * abar_k`); it does not
  hold for `'offset'`'s `m`/`e`/`ebar_k` decomposition, so a T-family run
  under `synthesis='offset'` is not a meaningful construction even though
  the code path doesn't block it.
- **J (Joint)** — dialed on the JOINT (Adequacy, Fluency) pair, via
  `donor_family_joint`/`joint_aspect_lookup_table` in
  `codes/lib/synthetic_scorers.py`. **J is restricted to `'offset'`** — it's
  the mirror-image restriction of T: it generalizes `'offset'`'s own
  `m`/`e`/`ebar_k` decomposition by conditioning `m` on the joint
  `(a(k,i), b(k,i))` pair (grouping cells that share BOTH aspect values
  exactly, via `np.unique(..., axis=0)`) instead of either aspect alone, so
  `m` retains whatever the donor's response to both aspects together looks
  like, and `ebar_k` (the dial term) removes only the per-system offset the
  joint doesn't already explain. Since `'additive'`/`'additive_mean'`
  inject the aspect directly rather than going through `m`/`e`/`ebar_k`,
  they have no joint analogue. Unlike A/B, J has no A-side/B-side split —
  a single dial family, since the joint conditioning already uses both
  aspects at once. Wired into `donor_family_spa_points`
  (`codes/lib/synthetic_scorers.py`, key `'J'`, only emitted when
  `synthesis='offset'`), and from there into
  `all_synthetic_family_points` — so an offset-synthesis donor now expands
  into `3 * len(dial_grid)` points instead of `2 * len(dial_grid)`. Not yet
  wired into the alpha-grid/orientation pipeline
  (`synthetic_scorer_alpha_grid.py`, `synthetic_scorer_orientation.py`) or
  the plotting scripts, which still only read `'A'`/`'B'`(/`'T'`).
- **AB (Adequacy+Fluency, two dials at once)** — dialed on BOTH aspects
  simultaneously, one independent dial per aspect, via
  `donor_family_additive_mean_ab` in `codes/lib/synthetic_scorers.py`.
  **AB is restricted to `'additive_mean'`** (no `'offset'`/`'additive'`
  analogue): it generalizes `donor_family_additive_mean`'s single-dial
  `y(k,i) + dial * abar_k` to two dials at once,

      y_dial(k,i) = y(k,i) + A * abar_k + B * bbar_k

  (`abar_k`/`bbar_k` the same per-system means `donor_family_additive_mean`
  uses, each independently standardized when `standardized=True`, the
  default). Unlike every other family, its sweep is 2-D — every `(A, B)`
  pair from a dial grid's own Cartesian square, `len(dial_grid)**2` points
  instead of `len(dial_grid)`, keyed by `(A, B)` tuples rather than a
  single float — which is why AB is wired into `donor_family_spa_points`
  (and `all_synthetic_family_points`'s passthrough) as **opt-in** via an
  `ab_dial_grid` parameter defaulting to `None` (AB omitted, so every
  existing caller's cost is unchanged) rather than built automatically the
  way A/B/T/J are. Not yet wired into the alpha-grid/orientation pipeline
  or the plotting scripts (same status as J above) — those assume a scalar
  float dial key throughout, so wiring AB in later needs to account for its
  tuple keys too, not just the quadratic cost.

## "Fake donor" aspect donors (`ASPECT_DONORS`)

`donor_alpha_dial_grid` also accepts `donor_name` values of `'AdequacyMQM'`,
`'FluencyMQM'`, `'AllMQM'` — instead of a real metric's segment scores, the
donor is one of the gold aspect signals itself (`a_pos`, `b_pos`,
`human_seg`), i.e. what happens if the base scorer IS a perfect oracle for
one aspect (or the total). Same family/alpha/metametric machinery
downstream as any real donor.

## Where this feeds into

- `codes/lib/synthetic_scorer_alpha_grid.py` — builds the (dial, alpha)
  weighted-meta-metric grid per donor per family (SPA or PA against All
  MQM), reweighting solved once per alpha via `lib.reweight_exact` and
  shared across donors/dials.
- `codes/lib/synthetic_scorer_orientation.py` — orientation-score analysis
  (`adequacy_orientation`, `fluency_orientation`, `allmqm_orientation`) over
  the families; caches are tagged by synthesis (`_synthesis_suffix`, no
  suffix for `'offset'` to keep pre-existing cache names valid) and dial
  preset.
- `codes/scripts/compute_scorer_orientation_vs_alpha.py`,
  `plot_scorer_orientation_vs_alpha.py`,
  `plot_allmqm_orientation_vs_ess_pooled.py`,
  `plot_synthetic_scorer_families.py` — drivers and plots, all exposing
  `--synthesis offset|additive|additive_mean` and `--dial-preset
  linear|geometric`.
