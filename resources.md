# Resources catalog

Resource gathering completed on 2026-10-04; experimental results are not yet available. Fresh isolated uv environment, lockfile and local data are ready. Start with [planning.md](planning.md) and [literature_review.md](literature_review.md).

## Papers

10 downloaded PDFs (all 9 specified arXiv papers plus SEP), and 1 specified HTML article.

|Title|Authors|Original year|File|Pages / version|
|---|---|---|---|---|
|The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems|Ren, Richard et al.|2025|[PDF](papers/2503.03750.pdf)|22 / 2503.03750v3|
|Liars&#39; Bench: Evaluating Lie Detectors for Language Models|Kretschmar, Kieron et al.|2025|[PDF](papers/2511.16035.pdf)|112 / 2511.16035v2|
|Detecting Strategic Deception Using Linear Probes|Goldowsky-Dill, Nicholas et al.|2025|[PDF](papers/2502.03407.pdf)|35 / 2502.03407v1|
|One Probe Won&#39;t Catch Them All: Towards Targeted Deception Detection|Natarajan, Vikram et al.|2026|[PDF](papers/2602.01425.pdf)|16 / 2602.01425v2|
|Probing the Limits of the Lie Detector Approach to LLM Deception|Thormann, Tom-Felix|2026|[PDF](papers/2603.10003.pdf)|10 / 2603.10003v1|
|Asymmetries in Spontaneous and Instructed Deception|Luikham, Josiah|2026|[PDF](papers/2609.00180.pdf)|22 / 2609.00180v1|
|Beyond Liars&#39; Bench: The Impact of Lie Typology, Depth, and Sparsity on Deception Detection in LLMs|Moustafa, Amr et al.|2026|[PDF](papers/2607.20479.pdf)|9 / 2607.20479v1|
|Do LLMs Really Know What They Don&#39;t Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness|Cheang, Chi Seng et al.|2025|[PDF](papers/2510.09033.pdf)|18 / 2510.09033v3|
|&#34;Did you lie?&#34; Evaluating Lie Detectors across Model Scale and Belief-Verified Model Organisms|Cooney, Alan et al.|2026|[PDF](papers/2606.12618.pdf)|32 / 2606.12618v2|
|Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs|Kossen, Jannik et al.|2024|[PDF](papers/2406.15927.pdf)|22 / 2406.15927v1|

Anthropic: [Fine-Tuned Lie Detectors Failed to Generalize](https://alignment.anthropic.com/2026/lie-detectors/), Hopkins, Khullar, Wang and Roger,2026; local HTML `notes/anthropic_lie_detectors.html`. Detailed bibliography: [papers/README.md](papers/README.md). Integrity and original metadata: `notes/paper_metadata.json`. Full sequential reading notes: `notes/reading_notes.md`; PDFs were chunked into six-page units and inspected using rendered sheets with text supplementation.

## Datasets

2 requested datasets, 9 validated Parquet files. See [datasets/README.md](datasets/README.md) for pinned download commands, loading and sample links.

|Name|Source|Downloaded size/rows|Location|Use and caveat|
|---|---|---|---|---|
|MASK|[HF](https://huggingface.co/datasets/cais/MASK)|0.84MB / 1,000|datasets/mask/|Six test configs;1,000 actual public examples; must elicit chosen model beliefs.|
|TriviaQA|[HF](https://huggingface.co/datasets/mandarjoshi/trivia_qa)|63.93MB / 173,538|datasets/triviaqa/|rc.nocontext all splits; deduplicate IDs/text; test gold answers unavailable.|

Licenses: MASK unspecified in HF metadata, TriviaQA unknown. Preserve provenance; do not infer dataset rights from software licenses. Metadata validation is not a factual audit. No raw dataset is tracked in outer git; three-record samples are retained. Model outputs/mechanism labels still need generation.

## Code repositories

5 root clones,  including 1 placeholder, and 1 additional pinned probe submodule. [code/README.md](code/README.md) documents dependencies, entrypoints and inspection limits.

|Name|Source|Location|Commit|
|---|---|---|---|
|mask|[GitHub](https://github.com/centerforaisafety/mask)|code/mask|25e0b1201e6c928ebe69f7c5aad6fa9063a377ea|
|apollo|[GitHub](https://github.com/ApolloResearch/deception-detection)|code/apollo|f8ec4010e74927394709dffa22b97bdf8cd5a62f|
|liars-bench|[GitHub](https://github.com/Cadenza-Labs/liars-bench)|code/liars-bench|ba10de150873d53a34e88278346f857962f82de3|
|semantic-entropy-probes|[GitHub](https://github.com/OATML/semantic-entropy-probes)|code/semantic-entropy-probes|02e2167dd1c00e27080d421f9b40e13e00f0452b|
|knowledge-recall|[GitHub](https://github.com/AndyCheang/knowledge-recall-vs-truthfulness)|code/knowledge-recall|8c9f81a431e187560ab3a1a1ce323c3638d5b14b|
|Cadenza probe submodule|[GitHub](https://github.com/Cadenza-Labs/deception-detection)|code/liars-bench/src/probes|bb4ed7fb51e6d9b7abc4b089d04c9f5566e7bc8a|

MASK provides belief/scoring machinery; Apollo and Cadenza provide white-box probe baselines; SEP provides uncertainty/accuracy probes. The knowledge-recall repository has no implementation yet. No end-to-end inference baseline was run. Bundled repository data are reference material, not additional independently validated datasets.

## Search and selection

Keywords: LLM lie detection; strategic deception; epistemic uncertainty; hallucination; confabulation; white-box linear probes; belief verification; knowledge recall; semantic entropy; detector generalization. The required diligent paper-finder was tried first and returned HTTP500 (`logs/paper_finder.json`). We switched to arXiv, publisher/author pages, official GitHub and Hugging Face; user references were handled before additional resources. All specified references resolved. SEP was added to supply the missing explicit hallucination baseline, rather than expanding the research directions. This is a focused narrative search, not an exhaustive systematic review or citation-count ranking.

`planning.md` records evidence/relevance/information/feasibility scores for eight plausible directions. Only three survive. Extra repositories linked in the literature were not cloned where they support pruned steering, SAE sweeps or non-false deception. No staged local resources were specified. No downloads failed in the final artifact set; the search-service failure and placeholder code are documented limitations.

## Experiment handoff

1. Use one fixed open-weight model; Qwen2.5-7B-Instruct is a feasible candidate given the visible A6000≈48GiB. Download/pin model and entailment revisions during experiment setup; weights are not staged here. Root Python3.12.8 environment has data/PDF tools only. Resolve compatible torch/Transformers/sklearn dependencies rather than installing all historical repo pins together.
2. Start 100-question feasibility pilot, then 2,000 deduplicated TriviaQA questions plus MASK external validation. Freeze group splits and independent knowledge checks before pressure generation. Separate stable wrong, uncertain wrong, known-answer misreporting and unresolved cases. Do not use detector output as label evidence.
3. Compare Apollo/follow-up probes with SEP/correctness probes and semantic entropy/log-likelihood controls. Hold out incentive families; include pressured honest negatives and answer/prompt shortcut controls. Current Apollo code already exposes response-level thresholds; the historical discrepancy is not a blanket bug claim about the downloaded revision.
4. Report distribution-specific classified false-output shares with unresolved bounds, paired incentive effects, L-vs-H AUROC/AUPRC, subgroup FPR, confusion matrices and question-cluster uncertainty. Do not turn a curated/balanced dataset into deployment prevalence claims.

## Reproducibility and validation

- `uv sync` recreates the fresh environment from root pyproject/lockfile.
- `.venv/bin/python scripts/download_datasets.py` downloads exact dataset revisions and validates Parquet.
- `.venv/bin/python scripts/clone_resources.py` restores pinned root repos and required probe submodule.
- `.venv/bin/python scripts/gather_papers.py` restores reviewed arXiv revisions using source URLs and SHA256 in `notes/paper_metadata.json`; fails explicitly if bytes differ.
- `.venv/bin/python scripts/validate_resources.py` verifies local PDFs/hashes, dataset counts/hashes, repo commits and git exclusions. Evidence: `notes/resource_validation.json`.

Unresolved: latent knowledge/intent identifiability, noisy aliases/ground truth, dataset license metadata, unknown pressure yield on the chosen model and untested model-generation dependencies. These constrain experiments; none is hidden by the completion marker.

## Experiment execution update — 2026-10-04

The resource-phase statements above are historical. Execution is now complete; see REPORT.md and README.md.

- Fixed Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28, selected as a historical white-box replication baseline; live OpenRouter catalog snapshots and pricing saved. Model weights downloaded to workspace .cache/huggingface.
-600 TriviaQA questions,355/124/121 entity-disjoint train/validation/test;11,400 real local generations,3,000 activation records,400 separate API responses.100-question MASK matched-persona adaptation added700 local generations.
- Workspace uv environment uses torch2.6.0 after torch2.14’s missing-compiler failure. Exact environment and source hashes are in results/environment.json and reproduction_manifest.json. A6000 peak allocated17.92GB; generation64/activation8 batches, BF16.
- Refit linear deception/error/answer-entropy probes, prompt-only and text controls. Full SEP and trained follow-up detectors were not replicated. Only2 incentive mismatches recovered in follow-up (1 test), so verified-mechanism learning/evaluation is infeasible at this sample size.
- Strict false-output mixture939 cases:2 recovered proxies,124 uncertain-neutral errors,245 repeated stable-wrong beliefs,568 unresolved. Reference-audited813 cases:2/109/195/507. These are operational labels, not genuine-intent or deployment prevalence estimates.
- Instructed false-versus-uncertain-error response AUROC.972, prompt-only1.000; reward known-correct flag rate12/53. High prompt confounding is the reliable diagnostic, not a universal detector-performance claim.
- Blinded equivalence audit used current catalog model qwen/qwen3.8-27b; all raw replies, one malformed-array failure family, and individual-answer repairs are archived. The auditor is fallible; qualitative candidate review is an agent review, not human adjudication.
- Deterministic analysis matched7/7 checked output files, greedy replay10/10, causal prompt-feature check passed, artifact validation20/20. Results and figures are saved; no external publication or communication occurred.
