# Completed manuscript review

## Resource review

- Read `.codex/skills/paper-writer/SKILL.md` and the full lab writing style guide before drafting.
- Inspected `paper_examples/`: only a README was present; no example content was copied.
- Verified the supplied command files and NeurIPS style; customized project macros.
- Saved the outline and evidence map to `paper/OUTLINE.md` before writing prose.
- Used the exact requested author line: `Xiaoyan Bai and NeuriCo`.

## Scientific and language review

- The title, abstract, and conclusion distinguish a measured prompt confound from unresolved verified-misreport detection.
- The abstract contains 190 whitespace-delimited words, within the 150–250 requirement.
- Claims identify the historical model, operational labels, conditional denominators, and uncertainty.
- The primary contrast's one recovered test positive and the malformed training example remain explicit.
- Candidate and instructed contrasts are not presented as substitutes for genuine-lie detection.
- Strict and audited results are separate; audited labels do not refit probes.
- The text distinguishes absence of evidence from equivalence and bootstrap precision from identification.
- The introduction follows motivation, gap, approach with method figure, quantitative preview, three contribution bullets, and organization.
- Related work is organized by theme and cites all 11 supplied research works; two resource citations bring the bibliography to 13 entries.
- Bibliography metadata comes from the archived source metadata and the saved Anthropic article; no authors or venues were invented.
- Tables are generated directly from saved JSON/CSV artifacts. Added average-precision and layer-sensitivity values also come from those artifacts.
- Exact prompts, scoring, counts, resource revisions, hardware, failures, and execution deviations appear in the appendices.

## Formatting and build review

- Required package lines, command imports, `plainnat`, and modular section inputs are present.
- All numbered section files contain their section commands. The abstract uses the NeurIPS abstract environment.
- Tables use booktabs, no vertical rules, edge-padding removal, and grouped headers where appropriate.
- Numerical AUROC maxima are bold, with an explicit warning that this does not constitute significant detector ranking.
- All four figures have self-contained captions; the main two are vector PDF figures.
- Fixed the supplied style's undefined `\@trackname` under `[final]` without changing the requested style option or claiming conference acceptance.
- Completed the requested pdflatex → bibtex → pdflatex → pdflatex sequence successfully after final edits.
- Final logs contain no LaTeX errors, undefined citations/references, overfull boxes, or rerun warnings. Some underfull-box notices remain as nonfatal spacing diagnostics.
- Final PDF: 16 pages (nine main text, one references, six appendix), 84 hyperlinks, 13 bibliography entries.
- Rendered and visually reviewed all 16 pages; separately inspected the main figures and dense table/bibliography pages.
- Enlarged main figure labels, removed overlapping diagram text, and allowed long artifact paths to break.
- Checked PDF text bounds and internal link destinations: no out-of-page text, invalid destinations, or unresolved `??` references.

Validation details are saved in `review/validation.json`. The study's scientific limitations remain limitations; successful paper compilation does not resolve them.
