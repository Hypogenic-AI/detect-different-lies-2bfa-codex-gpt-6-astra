# Paper outline and evidence map

Title: Prompt Sensitivity and Sparse Verified Misreports in Matched-Question Deception Probing
Author: Xiaoyan Bai and NeuriCo

## Abstract
- Motivation: separating uncertain falsehood from incentive-associated misreporting.
- Fixed historical Qwen2.5-7B, 600 questions, 11,400 generations.
- Two recovered proxies; 60.49% unresolved; instructed AUROC .972 and prompt-only 1.000.
- One recovered test positive prevents verified specificity estimation.

## Introduction
- A false answer does not establish dishonesty; monitoring errors matter.
- MASK and belief verification already separate honesty and accuracy.
- Incremental matched-question diagnostic with independent knowledge and entropy samples.
- Method figure: sample/split, evidence, five arms/follow-ups, labels and frozen probes.
- Contributions: conservative composition, specificity controls, audit/transfer limitations.

## Related work
- Honesty, knowledge, belief verification: Ren, Cooney, Cheang.
- Activation probes and transfer: Goldowsky-Dill, Kretschmar, Natarajan, Luikham, Moustafa.
- Uncertainty and scope: Kossen, Thormann; calibration and audit: Hopkins.
- Sources: supplied literature review, local PDFs, notes/paper_metadata.json.

## Methodology
- Define knowledge strata and four-way output partition, formulas and estimands.
- Dataset revisions, entity splits, exact generation count accounting.
- Five arms, universal follow-ups, fictional incentives and fixed order.
- Strict scoring and blinded audit; no audit-driven probe refitting.
- Linear probe features/training, independent entropy approximation, calibration and inference.
- Evidence: src/common.py, results/prompts.json, summary.json, configuration and report.

## Results
- Arm table with confidence intervals and paired tests (summary.json).
- Composition strict/audited table; ambiguity range (audited_summary.json).
- AUROC table with class counts; prompt-only comparison and response plot (detector_metrics.csv).
- Known-correct subgroup false alarms table; interpretation limited to operational labels.
- Knowledge thresholds, audited robustness, API and adapted MASK.

## Discussion and conclusion
- Prompt sensitivity is established in this protocol; latent mechanisms are not identified.
- Small conditional samples, belief/follow-up intervention, historical model, calibration limits.
- Fallible scoring, ambiguous fragment, no independent human adjudication.
- Implications for monitoring and adequate-yield future studies; no deployment prevalence claim.

## Appendix
- Exact prompts and label definitions; hardware/revisions and count accounting.
- Full audit counts and paired tests; layer/regularization and answer-length sensitivities.
- External checks, candidate caveat and reproducibility/deviations.

## Presentation and checks
- Required NeurIPS preamble and literal author line; modular sections and booktabs tables.
- Abstract uses the standard abstract environment rather than a numbered section.
- Bold numeric AUROC maxima for scanning, explicitly not statistically significant rankings.
- No example papers available: paper_examples contains only README.md.
- Compile with pdflatex/bibtex/pdflatex/pdflatex; inspect all PDF pages and logs.
