# Stage 2A: parameterized experience boundary data

The first run is complete. See the [internal-validation results](STAGE2A_RESULTS.md)
for all 179 counterfactual states and the comparison with the task-type baseline.

Stage 2A samples **ALFWorld Train only**, freezes the existing compiler/backbone
and adapters, and measures the return difference of real same-state On/Off
continuations. Candidate rules nominate states; they never assign benefit labels.

## Predeclared partitions and provenance

The first controlled run uses six task types: Heat, Cool, Clean, Pick & Place,
Pick Two & Place, and Examine in Light. Per type it samples four gate-training,
two tuning, two internal-validation, and two sealed-test games: 60 total games.
Each type has at least two scenes in each partition. All partitions have disjoint
game paths, task families (all alternate trials), and scenes. Sliced/movable
variants are excluded in this initial run and recorded as a scope limitation.
The previously reserved scenes 6, 11, and 21 cannot enter gate training, tuning,
or internal validation. A reserved scene already inspected during setup is also
excluded from the sealed test; no inspected scene is advertised as untouched.

Only 48 games are executed; the 12 sealed games are listed but never reset,
collected, tuned on, or evaluated. Previously inspected scenes are excluded from
the sealed partition. This is a future gate test reserve, **not a certification
that the compiler or backbone never encountered those tasks**.

`overlap_audit.json` records available source hashes, ALFWorld-style IFT rows,
source trial identifiers, matching normalized skill texts and task descriptions,
and missing provenance. Available IFT data contains 8,700 records, including
7,486 ALFWorld-style records, but no trial IDs. The deployed skill texts have
confirmed matches in IFT contexts. Local pretraining `train.jsonl` is missing;
`val.jsonl` is available. Shared goal wording does not establish an identical
game, and missing IDs do not establish non-overlap. Original source manifests
are still needed to resolve compiler exposure.

## Real execution and snapshots

Each executed game first receives two full free-generation trajectories: Base
and the corresponding LoRA. Both use the native loader, official ALFWorld prompt
templates, five-step history, thinking chat template, tag-based parser, 4096
token generation/context limits, and 50-step episode budget. No new action
constraints or forced empty thinking are introduced. Pick Two uses the official
`pick_and_place` adapter mapping.

The implementation nominates up to four distinct pre-decision states per game:
key/transform-action opportunities, location interventions near those
opportunities, behavioral divergence, and recovery after invalid/repeated
actions. Initial-state controls fill available candidate slots. Candidate kinds
are heuristic metadata and never features or target labels. The initial dataset
does not promise balanced label classes; observed neutral/negative/positive
counts are reported without relabeling or fabricating examples.

Every snapshot records the exact game hash, complete action replay, observation,
inventory, full admissible commands, recent history, and replay-observation hash.
Each branch starts from a fresh simulator and identical replay. An inventory
query is applied identically to both branches as an observation probe. Snapshot
equality and initial tokenized-prompt equality are mandatory. These checks verify
deterministic replay; they do not rely on copying an opaque environment object.

For prefix length `t`, **both branches receive `50 - t` policy steps**; intervention
actions also consume prefix budget. The On branch keeps the frozen LoRA enabled
for its continuation; Off keeps it disabled. Target:

```text
R(s, mode) = 1 if the simulator reports task success within the remaining budget
Delta R(s) = R(s, On) - R(s, Off), in {-1, 0, +1}
```

No unique expert next action, task type, applicability rule, or source-trajectory
win supplies this target. Binary-return ties can contain different efficiency or
partial progress. This estimates the value of choosing a fixed continuation at
a state; it does not yet measure a gate switching repeatedly within an episode.

## Gates and evaluation

- **Task-Type Gate:** enable a type only if its family-balanced training return
  difference is positive. Type decisions never use validation outcomes.
- **Semantic Router:** TF-IDF cosine relevance between the current observable
  state and corresponding skill text; representation fits training data and skill
  texts only. The activation threshold is selected on tuning returns. This is a
  lightweight relevance baseline, not an oracle precondition classifier.
- **Benefit Gate:** TF-IDF plus three-class logistic regression predicts
  `P(positive) - P(negative)`. Fit on training counterfactual labels; select
  `C in {0.1, 1, 10}` and the activation threshold using tuning returns only.

State features whitelist task, current observation/inventory, recent observations
and actions, and available commands. They exclude game IDs, scene IDs, proposal
rules, replay identifiers, labels, future traces, and branch returns. Exact
cross-partition feature duplicates are removed by a predeclared train-first
rule without consulting rewards. Families receive equal fitting weight.

One internal-validation report compares these gates with Always Off, Always On,
and a retrospective branch oracle. It reports remaining-success rate,
family-balanced return, negative transfer, and positive-transfer retention.
Snapshots from one game are correlated; these are **not independent full-task
evaluations of a deployed dynamic gate**. Sealed-test execution belongs to a
later predeclared evaluation after the entire procedure is fixed.

## Run and artifacts

Use `conda activate latentskill` from the repository root:

```bash
python -m evals.boundary_gate.stage2a_data \
  --output-dir outputs/alfworld_stage2a_NEW_RUN
python -u -m evals.boundary_gate.stage2a_collect launch \
  --output-dir outputs/alfworld_stage2a_NEW_RUN --gpus 2,3,4,5,6,7
python -m unittest discover -s tests -v
```

Choose a fresh directory. Collection automatically runs gate training only after
every worker finishes successfully. Outputs include `plan.json`,
`overlap_audit.json`, per-worker model audits, discovery trajectories, complete
counterfactual traces in `outcomes.jsonl`, job/progress metadata, and
`gates/report.json` with saved gate artifacts and validation predictions. Failures
stop training rather than silently accepting a partial dataset. Load pickle gate
artifacts only from trusted local experiments.

The active run is `outputs/alfworld_stage2a_train_v3`. Earlier `v1`/`v2` manifests
were superseded before collection to strengthen scene diversity and preserve
earlier reserves. Their reset-only provenance checks are treated as prior scene
inspection when constructing the active sealed reserve. The separate one-step
infrastructure smoke is not gate-training data.
