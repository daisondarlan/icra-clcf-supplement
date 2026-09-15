# Landing-Site Feature Dataset

3,458 labeled rows of geometric landing-site features, from an Intel RealSense
D455 depth camera, across 8 sessions in two collection phases. Released as
supplementary material for the paper "The Cross-Layer Consistency Audit."

## Columns

| Column | Type | Units | Description |
|---|---|---|---|
| `session_id` | string | -- | One of A-E (Phase 1) or F_flat/G_tilt/H_clutter (Phase 2) |
| `collection_phase` | string | -- | `outdoor_phase1` or `controlled_phase2` (see below) |
| `source_file` | string | -- | Anonymized per-session CSV filename, retained for session-level traceability |
| `timestamp` | string | -- | Frame capture time, `YYYY-MM-DD HH:MM:SS.mmm` |
| `altitude_m` | float | metres | Camera height above the ground patch, estimated per-frame |
| `angle_deg` | float | degrees | Ground-plane tilt, least-squares plane fit in metric coordinates |
| `flatness_std` | float | metres | Standard deviation of depth within the region of interest |
| `obstacle_ratio` | float | fraction, 0-1 | Fraction of the region of interest exceeding the obstacle-height threshold |
| `dev_p50`, `dev_p75`, `dev_p90`, `dev_p95`, `dev_p99` | float | metres | Percentiles of \|depth - mean depth\| within the region of interest |
| `label` | int | -- | 1 = Landable, 0 = Not Landable |

## Collection phases

**Phase 1 (sessions A-E, outdoor).** Collected during real outdoor flights,
logged by an operator-gated capture tool: an operator confirms sensor
settling before logging starts, so no frame is logged before the sensor has
stabilized. This phase used an earlier version of the tilt-feature
computation, since corrected (see the paper, Sec. III-D); `angle_deg` values
in this phase reflect the pre-fix computation and should not be pooled with
Phase 2's `angle_deg` values without accounting for that.

**Phase 2 (sessions F_flat/G_tilt/H_clutter, controlled).** Collected after
the tilt-feature fix, using the same capture tool and depth camera in a
controlled setup built specifically to isolate two hazard types the outdoor
sessions did not cleanly separate: a propped-board tilt hazard and a clutter
hazard on an otherwise flat surface. This phase provides the clean,
independently verified ground truth used for the paper's leave-one-session-out
evaluation.

Every session was checked against the physically expected feature range for a
1.524 x 1.524 m ground patch before being included; one discarded session
(mean `obstacle_ratio` = 0.47 on a flat floor) is not in this release.

## Known limitation

Sessions A-D show an `obstacle_ratio` distribution inconsistent with a flat,
safe surface (mean 0.79-0.92 regardless of label), most likely caused by a
fixed obstacle-height threshold that does not hold across all outdoor ground
textures (see the paper, Sec. IV-C). Session E and both Phase 2 sessions do
not show this signature. This release includes the affected sessions
unmodified, with this flag, rather than silently dropping them -- the
inconsistency is itself part of the paper's evaluation.

## License

Released for research use alongside the paper. Contact the authors for other
uses.
