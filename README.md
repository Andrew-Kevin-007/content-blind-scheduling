# Prefill Dominates: Content-Blind Work-Aware Scheduling for LLM Inference Serving

Submission to **ICCCIoT-2026 / IC3IoT-2026**, Track 4 (Blockchain, Cloud
Computing and Big Data Analytics). Paper: `paper/main.pdf` (6 pages, IEEE
conference format, anonymized for double-blind review).

## Claim

In Microsoft Azure's production LLM inference traces, **prefill accounts for
91.2%–98.7% of schedulable work**, and prefill cost is exactly observable at
admission from the context length. A scheduler that never reads prompt content
therefore recovers **78.5%–92.5%** of the improvement attainable by an oracle
with perfect knowledge of output length, and **outperforms** a scheduler driven
by an explicit learned length predictor.

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python src/fetch_data.py     # ~1.8 GB, resumable, byte-size verified
python src/data.py           # build cache, print content digests
python src/run.py            # 300 simulations (~30 s on 8 workers)
python src/aggregate.py      # summary.csv, paired_tests.csv, headline.json
python src/sensitivity.py    # 150 simulations, cost-model robustness sweep
python src/figures.py        # figures/*.pdf
python src/tables.py         # paper/tables/*.tex
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Total runtime is a few minutes of simulation plus predictor training; no GPU is
required.

### Data

Azure LLM Inference Trace 2024, from
<https://github.com/Azure/AzurePublicDataset> (CC-BY-4.0 / MIT, no
authentication). `fetch_data.py` verifies exact byte counts, because a truncated
CSV still parses and would silently corrupt every downstream number — this
actually happened during development.

## Discipline

These rules are what make the numbers trustworthy; they are not decoration.

- **No hand-transcribed numbers.** Results flow
  `run.py` → `results/raw/*.json` → `aggregate.py` → `results/*.csv|json` →
  `tables.py` → `paper/tables/*.tex` → `\input` into `main.tex`. The abstract
  cannot drift from the tables.
- **Determinism is checked, not assumed.** `data.py` prints SHA-256 digests of
  every window; reruns must match. They do.
- **No label leakage.** `features.assert_content_blind()` fails the run if any
  feature is >0.95 rank-correlated with the target. All rolling statistics are
  shifted by one request so nothing sees its own or a future outcome.
- **Train/eval separation.** The predictor trains on the first 24 h of each
  trace; evaluation windows are asserted to start after that cut.
- **Paired comparisons.** The same five windows are replayed under every policy.

## Layout

```
src/config.py       every constant; seeds and parameters live here only
src/fetch_data.py   resumable, size-verified download
src/data.py         trace cache, deterministic window selection, digests
src/features.py     content-blind features + leakage guard
src/predictor.py    output-length predictor and its evaluation
src/simulator.py    continuous-batching replica, KV-bounded, iteration-level
src/run.py          experiment matrix
src/sensitivity.py  cost-model robustness sweep
src/aggregate.py    the only path from raw runs to reported numbers
src/figures.py      figures (greyscale-safe: marker + linestyle, not colour)
src/tables.py       LaTeX tables
paper/              manuscript, bundled IEEEtran.cls/.bst, generated tables
results/            raw run records and derived summaries (committed on purpose)
```

## Submission checklist

- [x] 6 pages, within limit
- [x] Zero Type 3 fonts, all fonts embedded (IEEE PDF eXpress prerequisite)
- [x] Zero overfull boxes, zero undefined references
- [x] Anonymized (`\blindreviewtrue`) — double-blind review
- [x] Every citation verified against a primary source
- [ ] Similarity check ≤20% (Turnitin/iThenticate) — run before upload
- [ ] Upload to <https://cmt3.research.microsoft.com/ICCCIOT2026>
      (the conference site's own footer links a **stale 2024 portal** — ignore it)
- [ ] Set `\blindreviewfalse` and fill the author block **only** for camera-ready
