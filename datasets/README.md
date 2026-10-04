# Downloaded datasets

Two requested datasets are locally available as pinned Parquet files. Raw data are ignored by git; three-record examples in `samples/` are each below 100KB. Integrity, schemas and null counts: `../notes/dataset_validation.json`; split-ID overlap: `../notes/dataset_overlap.json`.

## Reproduce downloads
From the repository root (verify `pwd`):
```bash
uv sync
.venv/bin/python scripts/download_datasets.py
```
The script downloads only the six MASK configs and TriviaQA `rc.nocontext` (all splits), checks Parquet parsing, and records SHA256. No multi-GB context archives are needed. All network/data paths are explicit; no authentication was required.

## MASK
Source: https://huggingface.co/datasets/cais/MASK
Revision: `4602b84dd9e2ca05c6e1eafbc14e556e908ac1bb`.
1,000 actual rows across six configurations, each a `test` split. This differs from the card’s 1,028 claim and paper’s 1,500 evaluation including private examples. These are prompts and target facts, not ready-made model-specific lie labels. Start with known_facts; provided_facts needs independent comprehension verification. Columns include task_id, system_prompt, user_prompt, proposition, ground_truth, type and canary. Belief elicitation fields differ across configs; provided_facts has none. Do not remove the benchmark canary or use the data for pretraining. Data license is not declared in retrieved HF metadata; code license does not establish dataset license.

## TriviaQA
Source: https://huggingface.co/datasets/mandarjoshi/trivia_qa
Revision: `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`; configuration `rc.nocontext`.
Original splits train 138,384/validation17,944/test17,210. Distinct question IDs:76,523/9,960/9,521. Repeated source rows must be grouped/deduplicated before splitting or bootstrapping; zero ID overlap across original splits. Some equal texts have distinct IDs and vice versa, so group normalized text and ID together. Test answers are ALL `<unk>` placeholders despite being non-null: use train/validation for supervised experiment splits; test only for unlabeled generation. Official HF metadata says license unknown.

Fields include question, question_id, question_source and nested answer (value, aliases, normalized_aliases); empty context structures are intentional. Alias sets can be overly broad (first example includes Grace Hegger for Sinclair Lewis), so normalized alias matching needs an audited adjudication policy. Do not equate every non-matching answer with an error; inspect dates/numbers and ambiguous questions. No source context or gold answer may enter neutral model prompts.

## Loading
```python
from datasets import load_dataset
mask = load_dataset("parquet", data_files="datasets/mask/known_facts/test-00000-of-00001.parquet", split="train", cache_dir="datasets/.cache")
trivia = load_dataset("parquet", data_files={"train":"datasets/triviaqa/rc.nocontext/train-00000-of-00001.parquet", "validation":"datasets/triviaqa/rc.nocontext/validation-00000-of-00001.parquet"}, cache_dir="datasets/.cache")
```
Here `split="train"` is the local loader’s default container name; MASK itself has only test data. Alternative online loading uses `load_dataset("cais/MASK", "known_facts", split="test", revision=REVISION, cache_dir="datasets/.cache")`; the HF MASK card’s `cais/hle` example is a copy error.

## Validated files
|File|Rows|Unique IDs|Bytes|
|---|---:|---:|---:|
|datasets/mask/continuations/test-00000-of-00001.parquet|176|176|179367|
|datasets/mask/disinformation/test-00000-of-00001.parquet|125|125|89336|
|datasets/mask/doubling_down_known_facts/test-00000-of-00001.parquet|120|120|135677|
|datasets/mask/known_facts/test-00000-of-00001.parquet|209|209|169898|
|datasets/mask/provided_facts/test-00000-of-00001.parquet|274|274|203028|
|datasets/mask/statistics/test-00000-of-00001.parquet|96|96|65576|
|datasets/triviaqa/rc.nocontext/test-00000-of-00001.parquet|17210|9521|1199699|
|datasets/triviaqa/rc.nocontext/train-00000-of-00001.parquet|138384|76523|55391498|
|datasets/triviaqa/rc.nocontext/validation-00000-of-00001.parquet|17944|9960|7335321|
