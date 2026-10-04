# Detect different lies

A matched-question study of factual errors, independent knowledge evidence, and incentive-associated misreporting in Qwen2.5-7B-Instruct. It includes real local hidden-state probes, OpenRouter calls, prompt-only controls, and an exploratory MASK transfer check. **The verified lie-versus-hallucination comparison remains underpowered.**

- 600 TriviaQA questions;11,400 local generations, plus400 API responses and700 MASK-adaptation generations.
- Only2 incentive mismatches passed the neutral-knowledge and private-recovery criteria; only1 was held out. These are proxies, not established intent.
- Of813 reference-audited false outputs in the four-arm mixture:0.25% recovered misreport proxies,13.41% uncertain neutral errors,23.99% stable wrong beliefs,62.36% unresolved.
- Instructed-versus-uncertain-error response-probe AUROC0.972 was accompanied by prompt-only AUROC1.000. Correct reward answers were flagged12/53 times at a neutral-calibrated threshold.
- Exact-score and model-audited sensitivity analyses, raw outputs, probe weights, and uncertainty intervals are retained.

Read [REPORT.md](REPORT.md) for methods, results, limitations and references.

## Reproduce analysis from cached outputs

Run from this workspace root; Python3.12 and uv are recommended.

```bash
pwd
uv sync --frozen
source .venv/bin/activate
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python src/analyze.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python src/audit_analysis.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python src/analyze_mask.py
python src/write_report.py
python src/validate.py
```

All answer audits are cached. `analyze_mask.py` makes API requests only if reference audits are missing. The saved determinism check compares two full analyses; validation also reads the saved real-model replay. Analysis takes approximately a minute on the available CPU, depending on BLAS threads. Existing results are overwritten deterministically; raw generations are not regenerated.

## Reproduce generation

A CUDA GPU with about24GB or more is recommended; the measured A6000 peak allocated memory was17.92GB. Allow roughly25GB for model weights and dependency storage beyond the existing environment, plus about0.2GB of research artifacts. Model downloads and fresh API routing can affect run time and reproducibility. The experiment used an isolated `.venv`, never conda or a shared environment.

Set `OPENROUTER_KEY` in the environment; do not put secrets in source files. The model is ungated. For a clean generation rerun, archive the existing results first: inference intentionally resumes from cached record IDs. Preserve `results/model_metadata.json`, `notes/live_model_catalog.json` and the pinned dataset files, or restore resources with the scripts described in [resources.md](resources.md).

```bash
pwd
source .venv/bin/activate
python src/prepare.py
python src/run_api.py
python src/run_local.py --n 100
python src/run_local.py --n 600
python src/audit_answers.py
python src/run_mask.py
python src/replay_checks.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python src/analyze.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python src/audit_analysis.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python src/analyze_mask.py
```

Local scripts pin the model revision and download to `.cache/huggingface`. Pilot/full generation and activations took about6 minutes total, excluding model loading and download; MASK added26 seconds. Actual APIs, downloads and CPU analysis add time. Fresh generation is not guaranteed bitwise identical across GPU, batch-layout or library changes. API model/provider revisions are not pinned; archived responses are the reproducible record. The private follow-up coverage was expanded after the original pilot; saved per-batch seeds are authoritative for exact replay of this run.

## Files

- `src/`: preparation, real inference, answer auditing, probe/statistical analysis, report generation, validation.
- `results/model_outputs/`: exact local/API prompts and responses, auditor replies and failures.
- `results/activations/`, `results/probes/`: model-specific features and fitted linear/text detectors.
- `results/*summary.json`, `*metrics.csv`, `subgroup_flags.csv`: measured results and confidence intervals.
- `results/error_analysis.md`, `targeted_candidate_review.csv`: qualitative review and label problems.
- `results/validation.json`, `determinism_check.json`, `replay_validation.json`: verification evidence.
- `figures/`: standalone PNG plots included in the report.
- `planning.md`, `STATE.md`, `literature_review.md`, `resources.md`: protocol, handoff and resource provenance.
- `uv.lock`, `pyproject.toml`: isolated reproducible dependencies.

The report distinguishes strict aliases from reference-audited scoring and instructed controls from incentive behavior. No result establishes latent intent or deployment-wide lie prevalence.
