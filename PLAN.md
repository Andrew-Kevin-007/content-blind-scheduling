# EXECUTION PLAN — ICCCIoT-2026 / IC3IoT-2026, Track 4

**Status:** ACTIVE — issued 2026-08-02 17:30 IST by MAESTRO
**Deadline:** 2026-08-04 (extended, confirmed)
**Internal submission target:** 2026-08-04 **12:00 IST** — everything after that is slack, not schedule.
**Track:** 4 — Blockchain, Cloud Computing and Big Data Analytics
**Portal:** https://cmt3.research.microsoft.com/ICCCIOT2026 *(the CFP page also links a stale 2024 CMT URL — ignore it)*

> This plan is written to be executed without further deliberation. Where a choice
> exists, the choice has been made. Deviations require a stated reason logged in
> `results/DECISIONS.md`.

---

## 0. ENVIRONMENT — VERIFIED, NOT ASSUMED

Every item below was executed and its output inspected on 2026-08-02 between 17:15 and 17:30 IST.
The single largest schedule risk in a 48-hour paper is discovering a broken toolchain at hour 40.
That risk is now **retired**.

| Component | Status | Evidence |
|---|---|---|
| Python | **3.14.0** | `python --version` |
| numpy / scipy | **2.5.1 / 1.18.0** | imported successfully |
| pandas | **3.0.5** | imported successfully |
| scikit-learn | **1.9.0** | imported successfully |
| matplotlib | **3.11.1** + Agg backend | imported, headless render OK |
| cp314 wheels | **All available** — no source builds needed | pip resolved `*-cp314-win_amd64.whl` for every package |
| MiKTeX / pdfTeX | **25.12 / 4.23** | `pdflatex --version` |
| `IEEEtran.cls` | **INSTALLED** | `kpsewhich` → `.../tex/latex/ieeetran/IEEEtran.cls` |
| IEEEtran compile | **PASS** — 1-page PDF, 110 KB | full smoke test: title block, abstract, keywords, `amsmath` display eq, `booktabs` float, `thebibliography` |
| **Font embedding** | **PASS — all 10 fonts `emb=yes`** | `pdffonts`: NimbusRomNo9L ×4 + CM ×6, all embedded + subsetted |
| bibtex / biber | **4.2 / 2.21** | both present |
| latexmk | **4.88** | present |
| CPU | **16 cores** | `os.cpu_count()` |
| GPU | none assumed | — |

**Locked artifacts:** `requirements.txt` (18 pinned packages, `pip freeze`).

### Environment gotchas already discovered (do not rediscover these)
1. **Never write `.tex` files via bash heredoc.** The first smoke test failed with
   `! Undefined control sequence. <argument> Dept\City` because the heredoc consumed
   LaTeX backslashes. **Always use the Write/Edit file tools for `.tex`.** This cost
   one compile cycle; it must not cost another.
2. MiKTeX prints `major issue: So far, you have not checked for MiKTeX updates` on
   every invocation. **This is noise, not an error.** Do not chase it. Do not run a
   MiKTeX update during the 48 hours — an interrupted package-database update is a
   self-inflicted catastrophic risk with zero upside before a deadline.
3. `pdflatex` returns exit code 0 even when it produces no PDF. **Never gate on exit
   code. Gate on `ls` of the `.pdf` and on the page count read from the PDF itself.**

---

## 1. HOUR-BY-HOUR SCHEDULE

**Budget reality:** 2026-08-02 17:30 → 2026-08-04 12:00 = **42.5 h wall clock**.
Minus 14.5 h sleep (non-negotiable — a paper written at hour 40 without sleep gets
desk-rejected for the errors fatigue causes) = **28 working hours** for one author.
Every hour below is allocated against that 28, not against 42.

### Legend
`GATE` = a hard stop. Do not proceed past a failed gate; execute the named fallback instead.

---

### SUNDAY 2 AUG

| Time (IST) | Phase | Activity | Gate |
|---|---|---|---|
| 17:15–17:30 | **P0 — Env lock** | ✅ **COMPLETE.** Toolchain verified, deps pinned. | ✅ **G0 PASSED** |
| 17:30–19:00 | **P1 — Topic lock** | Receive zeus's ranked recommendation. Pick **one** topic. Download the dataset **immediately** — before any further planning. Verify row/class counts against the source's documented figures. Commit raw data checksum. | **G1: dataset on local disk, loads in pandas, shape matches the published description.** |
| 19:00–20:00 | **P1b — Scope freeze** | Write `results/DECISIONS.md`: problem statement (3 sentences), the exact claim the paper will make, baselines, metrics, seed count. **Freeze it.** | **G1b: one-sentence claim written down.** If it cannot be written in one sentence, the topic is not ready — invoke Risk R6. |
| 20:00–21:30 | **P2 — Data pipeline** | `src/data.py`: load → clean → **fixed** stratified split → cache to `data/processed/`. Splits written to disk once and never regenerated. | **G2: `python -m src.data` twice produces byte-identical split files.** |
| 21:30–23:30 | **P2b — Baseline harness** | `src/run.py`: full experiment loop, all baselines, **1 seed only**, on a 10% data subsample. Purpose is to prove the pipeline runs end-to-end, not to get results. | **G2b: `results/raw/` contains a valid JSON result row for every (method × seed) cell.** |
| 23:30 | **SLEEP** | Hard stop. Commit WIP first with an honest message. | — |

> **19:00 is the real decision point.** If zeus has not delivered by 19:00, do not wait.
> Take the highest-ranked candidate available and lock it. A locked mediocre topic beats
> an unlocked excellent one by a wide margin at T-42h.

---

### MONDAY 3 AUG

| Time (IST) | Phase | Activity | Gate |
|---|---|---|---|
| 07:00–07:30 | Restart | Coffee. Re-read `DECISIONS.md`. Do **not** change the scope. | — |
| 07:30–11:00 | **P3 — Full experiments** | Full data, **all methods × 5 seeds**. Launch as a background run. While it runs, write Section III (Methodology) — it does not depend on results. | **G3: every cell in the result matrix populated, zero exceptions in the log.** |
| 11:00–12:00 | **P3b — Ablation** | The single most informative ablation only. One variable. 5 seeds. | Soft gate — droppable (NICE-TO-HAVE). |
| 12:00–13:00 | **P4 — RESULTS FREEZE** | 🔒 `src/aggregate.py` → `results/summary.csv` + `results/metrics.json`. Paired t-test / Wilcoxon across seeds. **After this point no experiment is re-run to get a nicer number.** | **G4: `metrics.json` exists and is committed. Git tag `results-freeze`.** |
| 13:00–15:00 | **P5 — Figures + tables** | `src/figures.py` → `figures/*.pdf` (vector). `src/tables.py` → `paper/tables/*.tex`. Both read **only** from `results/`. Greyscale-safe from the first draft. | **G5: every figure regenerates from `results/` with one command; all are `.pdf` vector.** |
| 15:00–16:00 | Break / meal | Protected. Do not skip. | — |
| 16:00–20:00 | **P6 — Core writing** | Nerd drafts Abstract, I. Intro, II. Related Work, IV. Results, V. Conclusion into nerd's scaffold at `paper/`. Numbers come **only** from `\input{}`-ed generated tables. | **G6: complete draft compiles to PDF.** |
| 20:00–21:00 | **P6b — Page-limit gate** | Compile. Read the page count **from the rendered PDF**, never from a build log. | **G6b: ≤ page limit.** Over → cut Related Work first, then the ablation, then figure count. Never cut the reproducibility statement or the limitations paragraph. |
| 21:00–23:00 | **P7 — Zeus claims audit** | Zeus performs a claims-vs-evidence audit: every claim in the text traced to a number in `results/`. Unsupported claims are struck or hedged. | **G7: written audit with a verdict per claim.** |
| 23:00 | **SLEEP** | Commit first. | — |

---

### TUESDAY 4 AUG

| Time (IST) | Phase | Activity | Gate |
|---|---|---|---|
| 06:00–08:00 | **P8 — Audit remediation** | Fix everything zeus flagged. Re-verify each number against `results/summary.csv` by hand. | **G8: zero open audit items.** |
| 08:00–09:00 | **P9 — Reference sanity** | Every citation resolved to a real DOI/arXiv ID. Any reference not personally verified is **deleted**, not guessed. | **G9: 100% of references verified. See §5.** |
| 09:00–10:00 | **P10 — Final QA** | Full §5 pre-submission checklist. Greyscale print test. | **G10: checklist 100% ticked.** |
| 10:00–11:00 | **P11 — PDF eXpress + CMT** | Generate compliant PDF. CMT account creation, metadata, abstract, author list, upload. | **G11: submission ID received from CMT.** |
| 11:00–12:00 | **SLACK #1** | Reserved for CMT friction (see R4). If unused, re-read the paper once, cold. | — |
| 12:00 → EOD | **SLACK #2** | **~12 h of unearned slack.** Do not plan into it. | — |

**On AoE:** many venues treat deadlines as Anywhere-on-Earth, which would extend to
~05:30 IST on 5 Aug. **This has not been confirmed for this venue and must not be
planned against.** Treat it as invisible. If a genuine emergency occurs on 4 Aug,
it is a bonus, not a budget.

---

## 2. DEFINITION OF DONE + MINIMUM VIABLE PAPER

### 2.1 Per-phase DONE

| Phase | DONE means |
|---|---|
| P1 Topic | Dataset on disk, loads, shape matches published description, license permits research use, URL recorded in `DECISIONS.md`. |
| P1b Scope | Claim written in one sentence. Baselines named. Metrics named. Seed count fixed at 5. |
| P2 Data | Split files byte-identical across two independent runs. Checksums in `results/checksums.txt`. |
| P3 Experiments | Every (method × seed) cell has a JSON row. Zero silent exception handlers. |
| P4 Freeze | `results/summary.csv` + `metrics.json` committed and git-tagged. Includes mean, std, and a significance test. |
| P5 Figures | Every figure/table regenerates from `results/` via one command. Vector PDF. Greyscale-legible. |
| P6 Draft | Compiles clean. Every number in prose traceable to `results/`. No `TODO`/`XXX`/`??` in the PDF. |
| P7 Audit | Zeus verdict per claim; every claim SUPPORTED or removed/hedged. |
| P10 QA | §5 checklist fully ticked with evidence, not assertion. |
| P11 Submit | CMT submission ID in hand, screenshot saved to `results/submission/`. |

### 2.2 MINIMUM VIABLE PAPER — the MUST-HAVE core

This is the smallest artifact that is still an **honest, reviewable contribution**.
Anything less should not be submitted.

**MUST-HAVE (cutting any of these means do not submit):**

1. **One real, public, citable dataset**, downloaded and processed by committed code.
   No synthetic-only results. No "data available on request."
2. **One clearly-stated problem** and **one falsifiable claim.**
3. **≥ 2 baselines**, at least one of which is the *obvious* thing a practitioner
   would actually do. A strawman baseline is the fastest route to rejection.
4. **5 seeds minimum**, with **mean ± std** reported for every number. A single-run
   number in a Track 4 paper is a reviewer's first target.
5. **One statistical significance test** (paired t-test or Wilcoxon signed-rank across
   seeds) supporting the headline claim — reported **whatever it says**.
6. **One primary results table** (`Table I`) — methods × metrics, mean ± std, best in bold.
7. **One system/method figure** — what was built. Reviewers orient on this first.
8. **One results figure** — the headline comparison.
9. **A Limitations paragraph.** Non-negotiable. Its absence reads as either naivety or
   concealment; its presence is the single cheapest credibility purchase in the paper.
10. **A Reproducibility Statement** (§4.4) with the public code URL.
11. **Complete, verified references** — every one real (§5).

**NICE-TO-HAVE (cut in this exact order under time pressure):**

| Cut order | Item | Why it is safe to cut |
|---|---|---|
| 1st | Third/fourth figure | Two figures carry the story. |
| 2nd | Hyperparameter sensitivity sweep | Strengthens, not load-bearing. |
| 3rd | Second ablation | One good ablation suffices. |
| 4th | Third baseline | Two credible baselines are defensible. |
| 5th | Extended Related Work (>12 refs) | 10–12 well-chosen refs are adequate at this venue. |
| 6th | Runtime/complexity analysis | Nice for Track 4, not required. |
| 7th | Qualitative examples | Purely additive. |

**NEVER CUT:** seeds → std → significance test → limitations → reproducibility
statement → reference verification. These are the integrity floor. A paper that
cuts these is not a shorter paper; it is a different and worse one.

---

## 3. RISK REGISTER

| ID | Risk | P | Impact | Pre-planned mitigation |
|---|---|---|---|---|
| R1 | Dataset inaccessible / dead link / login wall | Med | **Fatal** | **Trigger: 19:00 Sun, G1.** Zeus is instructed to supply **3** candidates with live URLs. Download the #1 dataset **within 30 min of topic lock** — before writing any other code. If it fails: fall to candidate #2 immediately, no renegotiation. Hard rule: **a dataset that is not on local disk by 20:00 Sunday is disqualified**, regardless of how good the topic is. |
| R2 | Experiments produce a null result | **High** | High | **This is planned for, not feared.** See §3.1 — full reframing protocol. |
| R3 | LaTeX toolchain missing/broken | ~~Med~~ | ~~High~~ | **RETIRED.** Verified 2026-08-02: IEEEtran compiles, all fonts embedded, bibtex+biber present. Residual: do **not** run MiKTeX updates during the 48 h. |
| R4 | CMT account / registration friction | Med | **Fatal** | **Create the CMT account TONIGHT (Sunday), not Tuesday.** Log in, locate the ICCCIOT2026 track, start a submission, save a draft with a placeholder title/abstract. This surfaces email-verification delays, conflict-of-interest forms, and subject-area requirements 40 h early instead of 40 min early. 1 h of Tuesday slack is reserved for this. Use the CMT3 URL above; the CFP's 2024 link is stale. |
| R5 | Results too weak to claim a contribution | Med | High | Reframe per §3.1. The fallback claim ladder: *outperforms* → *competitive with, at lower cost* → *we characterise when X fails* → *we provide a controlled benchmark*. Every rung is honest and publishable; only the first requires a win. |
| R6 | Topic itself is not viable (zeus finds no novelty gap) | Low | **Fatal** | If zeus reports all 3 candidates are crowded, **do not chase novelty**. Pivot to a *rigorous empirical comparison/replication under a unified protocol* — a legitimate, reviewable contribution that is far more achievable in 28 h than genuine algorithmic novelty. State it as such in the abstract. |
| R7 | Compute blowup — experiments do not finish | Med | High | **Hard rule set at G2b: if one (method × seed) cell exceeds 10 min on 16 cores, the scope is wrong — subsample the data or shrink the model until it fits.** 5 seeds × 4 methods × 10 min ≈ 3.3 h, which fits the P3 window. Enforce with a per-run timeout in `src/run.py`. Checkpoint after every cell so a crash never costs more than one cell. |
| R8 | Page limit exceeded late | Med | Med | Page count checked at **G6b (Mon 20:00)**, not Tuesday. Cut order is pre-decided (§2.2). Content is cut; hedges, CIs, limitations, and the reproducibility statement are **never** cut to save space. |
| R9 | Author illness / laptop failure | Low | **Fatal** | Push to a private remote **tonight**. Commit at every phase boundary (§4.5). A dead laptop must cost hours, not the paper. |
| R10 | Fabricated/hallucinated citation reaches the PDF | Med | **Career-fatal** | §5 verification protocol. Zero tolerance. This paper supports MS admissions — a fake citation discovered later is materially worse than not submitting at all. |
| R11 | Similarity/plagiarism flag | Low | **Fatal** | Related Work written from personally-read abstracts, never paraphrase-chained. Boilerplate ("In recent years…") minimised. Check before submission (§5). |

### 3.1 NULL RESULT PROTOCOL — how to reframe honestly

**Trigger:** at G4 (Mon 12:00), the proposed method does not beat baselines, or the
significance test returns p > 0.05.

**Do not:** re-run with new seeds and keep the good ones · drop the losing baseline ·
tune only the proposed method · switch metric after seeing results · move the test set ·
quietly delete the significance test · report a single lucky seed.
Each of these is fabrication. Each is detectable. Each would destroy the exact credibility
this paper exists to build.

**Do — pick the first rung that the evidence actually supports:**

1. **Cost/efficiency pivot.** Accuracy is a statistical tie, but the method is faster,
   cheaper, or lighter. Retitle around the axis that *did* move. Report the tie plainly:
   *"within 0.4 points of the strongest baseline (p = 0.31, n.s.) at 3.2× lower inference
   cost."* This is a genuine Track 4 contribution — cloud/big-data reviewers care about cost.
2. **Conditional-win characterisation.** It wins in an identifiable regime (class imbalance,
   small data, high dimensionality). Reframe to *"we characterise the regime in which X
   helps"* and show the crossover. Often **more** interesting than an unconditional win.
3. **Controlled benchmark / replication.** Nothing wins. Reframe to a rigorous
   seed-controlled comparison under a unified protocol, and report that a widely-claimed
   advantage does not reproduce here. **Negative results are publishable when the protocol
   is strong** — and protocol strength is exactly what §4 buys.
4. **Honest limitation.** If even the benchmark framing is thin, say so in Limitations and
   submit the weaker, honest paper.

**Rule:** the title and abstract are rewritten to match the evidence. The evidence is
never rewritten to match the title. If the claim changes, `DECISIONS.md` records the
change, when, and why — that log is itself evidence of integrity.

---

## 4. REPOSITORY LAYOUT & REPRODUCIBILITY DISCIPLINE

### 4.1 Layout

```
D:\Research-IEEE\
├─ PLAN.md                  # this file
├─ README.md                # what/how to reproduce (written at P10)
├─ requirements.txt         # PINNED, generated by pip freeze  [DONE]
├─ .venv/                   # gitignored
├─ data/
│  ├─ raw/                  # AS DOWNLOADED. Never edited. Gitignored if large.
│  ├─ processed/            # deterministic outputs of src/data.py. Gitignored.
│  └─ SOURCES.md            # URL, access date, license, citation for every dataset
├─ src/
│  ├─ config.py             # ALL hyperparameters + SEEDS = [0,1,2,3,4]. Single source of truth.
│  ├─ data.py               # download → clean → split → cache
│  ├─ models.py             # baselines + proposed method, one interface
│  ├─ run.py                # experiment loop → results/raw/*.json (one file per cell)
│  ├─ aggregate.py          # results/raw/* → summary.csv + metrics.json + significance
│  ├─ figures.py            # results/ → figures/*.pdf
│  └─ tables.py             # results/ → paper/tables/*.tex
├─ results/
│  ├─ raw/                  # one JSON per (method, seed). Append-only.
│  ├─ summary.csv           # aggregated mean/std        [FROZEN AT G4]
│  ├─ metrics.json          # every number cited in the paper, by key
│  ├─ checksums.txt         # SHA256 of raw data + splits
│  ├─ DECISIONS.md          # scope, claim, every deviation with timestamp+reason
│  └─ env.txt               # python version, package versions, CPU, OS, date
├─ figures/                 # generated ONLY by src/figures.py. Never hand-edited.
└─ paper/                   # nerd's IEEEtran scaffold
   ├─ main.tex
   ├─ refs.bib
   └─ tables/               # generated ONLY by src/tables.py
```

### 4.2 Seeding & determinism

- **`src/config.py` holds `SEEDS = [0, 1, 2, 3, 4]`.** Nothing else defines a seed.
- Every stochastic call takes `random_state=seed` **explicitly**. Never rely on global state.
- Set at process start: `PYTHONHASHSEED=0`, `random.seed(s)`, `np.random.seed(s)`,
  and for any threaded BLAS, pin `OMP_NUM_THREADS` so numerics do not drift with load.
- **The split is generated once and cached.** Every run reads the same split files.
  Regenerating a split per run is the most common silent reproducibility break.
- **G2 enforcement:** run `src/data.py` twice, compare SHA256. Not equal → fix before proceeding.
- `src/run.py` writes `results/env.txt` (versions, CPU, timestamp, git commit hash) on
  every invocation.

### 4.3 How results reach the paper — the no-transcription rule

**A number is never typed into the `.tex` by hand.** Hand-transcription is where
"the abstract says 87.4 but Table I says 86.9" comes from, and reviewers do check.

```
src/run.py  →  results/raw/*.json
                    ↓ src/aggregate.py
            results/summary.csv + metrics.json      ← FROZEN AT G4
                    ↓ src/tables.py            ↓ src/figures.py
          paper/tables/main_results.tex     figures/*.pdf
                    ↓ \input{}                 ↓ \includegraphics{}
                        paper/main.tex
```

- Tables enter via `\input{tables/main_results.tex}`.
- Inline numbers in prose (abstract, results text) are drawn from `metrics.json` and
  **cross-checked by hand against `summary.csv` at G8.** Anything in prose must exist
  in `metrics.json`.
- **Rebuilding everything is one command**, and it is run once more at G10:
  `python -m src.run && python -m src.aggregate && python -m src.figures && python -m src.tables && latexmk -pdf paper/main.tex`

### 4.4 Reproducibility statement (goes in the paper)

Draft now, finalise at P10. German admissions committees and reviewers both read this.

> **Reproducibility.** All code, configuration, and analysis scripts required to
> reproduce every result in this paper are publicly available at `<URL>`. Experiments
> use the public `<DATASET>` dataset (`<URL>`, accessed 2026-08-0X). All results are
> reported as mean ± standard deviation over 5 fixed seeds (0–4) specified in
> `src/config.py`; data splits are generated once, cached, and checksummed.
> Dependencies are pinned in `requirements.txt` (Python 3.14.0). Experiments run on
> CPU only (16 cores) and complete in under `<N>` minutes end to end. Statistical
> comparisons use a `<paired t-test / Wilcoxon signed-rank test>` across seeds.

Include the exact one-command rebuild line in `README.md`.

### 4.5 Git discipline

Agents and sessions die mid-task; a lost stage is an unrecoverable hour.
**Commit at every phase boundary**, with honest WIP messages:

```
git commit -m "P2 DONE: data pipeline, splits deterministic (G2 pass)"
git commit -m "P3 WIP: 3/5 seeds complete, NOT AGGREGATED"
git tag results-freeze     # at G4 — nothing after this changes a number
```

- First commit **tonight**, plus a private remote (R9).
- Never commit `data/raw` if large; commit `checksums.txt` instead.
- `results/raw/` **is** committed — it is the paper's evidence base.
- A dishonest commit message ("done" when it is not) is how a broken intermediate gets
  mistaken for shippable. Say `OVER LIMIT`, say `NOT AGGREGATED`.

---

## 5. PRE-SUBMISSION QUALITY GATES (G10)

Tick with **evidence**, not memory. Nothing here is satisfied by "I think so."

**Format & compliance**
- [ ] Page count read **from the rendered PDF** ≤ the limit confirmed by nerd. Overlength fees understood.
- [ ] `\documentclass[conference]{IEEEtran}` — not `journal`, not `compsoc`. No margin/font/spacing hacks.
- [ ] **All fonts embedded + subsetted** — `pdffonts main.pdf`, every row `emb=yes`. *(Verified achievable: smoke test showed 10/10 embedded.)*
- [ ] PDF opens cleanly in a viewer other than the one it was built with.
- [ ] IEEE PDF eXpress: run if the venue provides a conference ID; otherwise the font/embedding check above is the substitute. Do not discover a PDF eXpress requirement on Tuesday — confirm from nerd's logistics findings Sunday night.
- [ ] No hyperlink/`hyperref` breakage; `\url{}` used for all URLs.

**Anonymisation**
- [ ] **Blind or not blind — confirm from nerd's findings and act accordingly.** Do not assume.
- [ ] If blind: no author names/affiliations/emails; acknowledgements removed; self-citations in third person ("Smith et al. showed", never "our previous work"); **code URL anonymised** (anonymous.4open.science) or replaced with "available upon acceptance"; **PDF metadata scrubbed** — author fields leak identity and are routinely missed.

**References — zero tolerance**
- [ ] **Every reference personally verified** to resolve to a real paper: DOI, arXiv ID, or publisher page opened.
- [ ] Author list, title, venue, and year match the actual publication.
- [ ] **Any reference that cannot be verified is DELETED, not guessed at.**
- [ ] Every entry in `refs.bib` is cited; every `\cite{}` resolves. No `[?]` in the PDF.
- [ ] Venue names consistently abbreviated per IEEE style.

**Figures & tables**
- [ ] All figures vector PDF (no rasterised text).
- [ ] **Greyscale test: print or convert to greyscale and read it.** Series distinguishable by marker/linestyle/pattern, not colour alone.
- [ ] Axis labels + units present; font in figures ≥ 8pt at final print size.
- [ ] Every figure and table referenced in the text and appears near its reference.
- [ ] Captions self-contained — readable without the body text.

**Scientific integrity**
- [ ] Every number in the abstract and prose matches `results/summary.csv`. Checked line by line.
- [ ] No claim exceeds the evidence; "outperforms" only where the significance test supports it.
- [ ] Mean ± std reported everywhere; significance test reported **including when it fails**.
- [ ] Limitations paragraph present and honest.
- [ ] Reproducibility statement present with a working URL.
- [ ] **No fabricated results, datasets, baselines, or statistics.** Zeus's G7 audit signed off.

**Originality**
- [ ] Similarity check run (Turnitin/iThenticate via institution, or a free checker on Related Work + Intro — the highest-risk sections).
- [ ] No text reused from another source without quotation and citation.
- [ ] Self-plagiarism checked if any prior report shares text.

**Submission mechanics**
- [ ] CMT account working, **created Sunday** (R4).
- [ ] Correct track selected: **Track 4 — Blockchain, Cloud Computing and Big Data Analytics.**
- [ ] Title/abstract in CMT **match the PDF exactly.**
- [ ] All authors added with correct affiliations and emails.
- [ ] Submission ID recorded; confirmation screenshot saved to `results/submission/`.
- [ ] Registration (05/09) and camera-ready (10/09) dates noted; **fees confirmed and budgeted before submitting** — an accepted paper that cannot be paid for is not indexed.

---

## 6. HONEST PROBABILITY ASSESSMENT

**P(submitting a complete, honest, reviewable paper by the deadline) ≈ 80%.**

**P(that paper being genuinely good — a clear contribution a reviewer would defend) ≈ 45–55%.**

**P(acceptance | submitted) ≈ 50–65%** — regional IEEE conferences of this type
typically have moderate acceptance rates, and a paper with real data, 5 seeds,
significance testing, and public code sits well above the median submission.

**Why 80% and not higher:** the topic is not yet locked at T-42h, and the dataset is
not yet on disk. Until G1 passes, the single largest source of variance is untouched.
**Why 80% and not lower:** the toolchain risk is fully retired, compute is adequate
(16 cores), scope is pre-cut, and the null-result path still produces a submittable paper —
meaning R2, historically the most common killer, cannot by itself sink this.

The honest caveat: 28 working hours is enough for a *solid, small, rigorous* paper.
It is not enough for an ambitious one. Attempting an ambitious one is the most likely
route to submitting nothing.

### The single biggest thing that would raise it

> **Constrain the topic at G1 to one whose dataset is downloadable in under 10 minutes
> and whose full 5-seed experiment matrix runs in under 3 hours on 16 CPU cores.**

Nothing else comes close. Scope chosen against *compute and data-access cost* — not
against how interesting it sounds — is the difference between a finished paper and a
half-finished one. A modest, complete, rigorously-executed contribution is publishable
and supports an MS application. An ambitious, half-finished one supports nothing.

**Operationally:** when zeus's ranking arrives, re-rank it by *time-to-first-result*
and pick the fastest candidate that still clears the novelty bar. If the top-ranked
topic needs a 40 GB download or a transformer fine-tune, **take the second one.**

---

## 7. IMMEDIATE NEXT ACTIONS

1. **Now:** commit this plan + `requirements.txt`; push to a private remote (R9).
2. **Tonight, before sleeping — do not defer:** create the CMT account and save a draft
   submission (R4). This is the cheapest catastrophic-risk removal available.
3. **On zeus's delivery (by 19:00):** lock the topic, re-ranked by time-to-first-result;
   download the dataset within 30 minutes; write `DECISIONS.md`.
4. **From nerd, confirm tonight:** page limit, blind vs. non-blind review, PDF eXpress
   requirement + conference ID, fees. These four change the checklist and must be known
   Sunday, not Tuesday.

---

*Deviation from this plan is permitted. Silent deviation is not — log it in
`results/DECISIONS.md` with a timestamp and a reason.*
