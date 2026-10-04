# Matched-question diagnostic paper

- `main.pdf`: compiled NeurIPS-style manuscript (nine main-text pages, references, and appendices).
- `main.tex`: document entry point; exact requested author line.
- `sections/`: abstract and six numbered main sections.
- `tables/`: standalone tables generated from saved result artifacts.
- `figures/`: vector protocol/detector figures and original experiment plots.
- `appendix/`: exact protocol, supplementary results, audit, and reproducibility details.
- `references.bib`: 13 references, including every paper in the supplied literature review.
- `REVIEW.md`: completed writing and technical review.

Build from this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

An additional LaTeX pass may be needed when changed float placements alter references. The source project includes all required figure assets and the supplied style file. Standard LaTeX packages provide its remaining dependencies.

To regenerate tables, bibliography, and figures from the original workspace:

```bash
../.venv/bin/python scripts/build_assets.py
```

This script reads `../results/`, `../notes/paper_metadata.json`, and `../figures/`. Compiling the already generated manuscript does not require those experiment artifacts or Python.

The source uses the requested `\usepackage[final]{neurips_2025}`. The supplied style leaves its track notice undefined under `[final]` alone; `commands/macros.tex` supplies a notice identifying this as an October 2026 research manuscript. This does not imply acceptance at NeurIPS 2025. The abstract follows the standard NeurIPS abstract environment; numbered sections each live in their own file.

`paper_examples/` contained only a README, so there were no example introductions, tables, or macros to inspect. Formatting follows the required lab guide and supplied command templates. The outline and evidence map are in `../paper/OUTLINE.md`.
