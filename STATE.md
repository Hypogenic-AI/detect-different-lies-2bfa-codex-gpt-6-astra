# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-04T07:45:54.301096Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/detect-different-lies-2bfa-codex-gpt-6-astra`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
Resource finder Phases1–5 complete (2026-10-04). Artifacts: literature_review.md, resources.md, planning.md; papers/datasets/code READMEs; reproducible scripts; .resource_finder_complete. Validation:100 checks passed in notes/resource_validation.json; documented HF local loader also succeeded.
- Literature:10 PDFs (all9 supplied arXiv papers + SEP), supplied Anthropic HTML; all PDF chunks read. Full notes: notes/reading_notes.md; hashes/revisions: notes/paper_metadata.json. Paper-finder HTTP500; primary-source fallback succeeded.
- Data: pinned MASK1,000 rows across6 configs; TriviaQA rc.nocontext173,538 rows including unlabeled test (<unk>). Use train/validation; deduplicate question ID/text, audit noisy aliases. Schemas/hashes/counts: notes/dataset_validation.json; zero original split-ID overlap. Raw data ignored by git, small samples retained. Data licenses unspecified/unknown.
- Code:5 root clones (knowledge-recall is README-only) + pinned Cadenza probe submodule. notes/code_manifest.json and code/README.md record commits, entrypoints and dependency conflicts. No baseline inference run. Current Apollo threshold code distinguishes response calibration from token ReLU offset; historical correction reported in Cooney was not reproduced here.
- Direction budget: eight scored, exactly three retained in planning.md: mechanism strata; detector specificity; held-out incentive/measurement robustness. No later expansion. Preserve unresolved labels, independent knowledge/entropy samples, pressured honest and confident wrong controls. Shares are operational and distribution-specific, not proof of intent or deployment prevalence.
Next: experiment_runner reads these artifacts, installs compatible minimal inference dependencies in root uv environment, pins one model (Qwen2.5-7B-Instruct candidate), runs100-question feasibility pilot on visible A6000≈48GiB, then freezes grouped splits/knowledge thresholds and implements only the three retained directions. Model/entailment weights not downloaded. Unknown pressure yield and latent-belief identifiability remain; no experimental shares claimed.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
Phase6 documentation and validation complete. REPORT.md and README.md contain actual results and reproduction instructions; resources.md updated. Main artifacts:11,400 local generations on600 TriviaQA questions;3,000 activations;400 API responses;700 generations on MASK100 adaptation. Strict false mixture939=2 recovered proxies+124 uncertain neutral+245 stable wrong+568 unresolved; audited813=2+109+195+507. Only1 verified test positive: central mechanism comparison underpowered, supervised diagnostic not fitted. Instructed false/H response AUROC.972 vs prompt-only1.000; correct reward flag12/53. Audit/agent-review caveats explicit. All20 validation checks pass,7/7 analysis files match across reruns,10/10 greedy replay, causal prompt difference0. No model training or published checkpoint replication claimed. Generated pipeline header preserved per contract. Final process check confirms no running experiment processes (results/process_check.json). No further research phase pending.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
