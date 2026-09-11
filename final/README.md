# ICCCIoT-2026 — Paper 548, final submission set

**Quantifying the Cost of Content-Blindness in LLM Inference Scheduling**
Kevin Andrew A, Kavita Sri — Dept. of Information Technology,
Loyola-ICAM College of Engineering and Technology, Chennai.

Assembled 10 Sep 2026. Everything here is verified, not assumed; the checks
are listed against each file.

---

## Deliverables

Upload exactly these three to CMT, with these exact filenames (per the
conference team's email of 11 Sep 2026):

| Upload as | File here | Verified |
|---|---|---|
| `548_CameraReady.pdf` | `548_CameraReady.pdf` | 6 pages · US Letter 612×792 pt · 20/20 fonts embedded · 0 Type 3 · 0 overfull boxes · 11/11 citations · 167/167 numeric claims traced to `results/` |
| `548_Abstract.pdf` | `548_Abstract.pdf` | 1 page · organisers' template · abstract extracted programmatically from `main.tex`, so it cannot drift |
| `548_CameraReady.docx` | `548_CameraReady.docx` | 6 pages in Word · two-column IEEE · equations as real text with true subscripts, not images · 1 embedded image (the figure) · 3 tables |

**File 3 is the Word document itself**, not the IEEE copyright form — confirmed
by the conference team. eCopyright is a separate, unrelated step.

### PDF / Word content equivalence

The team requires both versions to carry the same final content. Checked
mechanically: all 28 headline figures appear in both, and the only numeric
differences are artefacts of extraction, not content — the PDF's Fig. 1 axis
and log-scale labels (0.60–0.98, 10²–10⁵) are selectable text in LaTeX but
pixels inside the embedded image in Word. Citations render `[9], [10]` in the
PDF and `[9, 10]` in Word; same references, both valid IEEE.

## Source

`source/` rebuilds `548-camera-ready.pdf` from scratch. The layout mirrors the
working tree because `main.tex` includes the figure as `../figures/`.

```
cd source/paper
pdflatex main.tex && pdflatex main.tex
```

Confirmed: this produces a 6-page, 286,669-byte PDF with the same two
(underfull, i.e. cosmetic) box warnings as the shipped build. PDF hashes differ
between runs only because pdfTeX embeds a build timestamp.

`source/Abstract-template-from-organisers.tex` is the unmodified template that
came with the acceptance email, kept for provenance.

---

## Reviewer #1 response (already in this build)

- Generalizability scoped explicitly: the claim is bounded to the measured
  regime, with its context-to-output ratios (15.5:1 and 110.7:1) stated and the
  16× sweep offered as evidence along that axis.
- `SLO`, `TTFT`, `TBT` now expanded at first use.

## Word CRC — repairs applied

The `.docx` is generated, not a PDF conversion. Four defect classes were found
and fixed in this copy:

1. **Algorithm 1** was mangled — lines 1, 2, 6, 7 were empty (the `on arrival`
   statement and `break` had been dropped), comments were orphaned, and literal
   `\{i\}` backslashes leaked. Rebuilt as 11 correctly numbered lines against
   the `algorithmic` source.
2. **10 leaked `\ref` label names** (`Section characterisation`,
   `Table recovery`, …) resolved to their real numbers from `main.aux`.
3. **8 orphan `\label{}` paragraphs** (`sec:intro`, `sec:setup`, …) were
   visible in the body; deleted.
4. **41 literal underscore subscripts** (`P_99`, `y_i`, `ρ_s`, …) converted to
   real Word subscript runs. Minus sign in eq. (3) corrected to U+2212.
5. **7 dropped hat accents in prose.** The generator lost `\hat{y}` and
   `\widehat{W}`, which made one sentence contradict itself — it read
   *"charges the true output length yᵢ, not the estimate yᵢ."* Restored; the docx
   now carries exactly 10 hat glyphs, matching the 10 hat macros in `main.tex`.

Verified after repair: 0 ref leaks, 0 underscore math, 0 orphan labels, 0 stray
braces, 0 literal backslashes, 10/10 hats, 1 image (the figure), 3 tables.

> Do **not** substitute a PDF→Word conversion here. The online conversion tried
> earlier rasterised maths into 22 sub-2 KB PNGs and dropped the `W` from
> equation (2). The acceptance letter forbids that: *"All mathematical equations
> must be prepared using an appropriate Equation Editor or MathType and should
> not be inserted as images or screenshots."*

**Not verified:** the `.docx` page count. Word COM automation is unreliable on
this machine (it failed three times). Open it in Word and check before
uploading if page count matters for that slot.

---

## Open items

- **IEEE eCopyright** — still not enabled by the organisers (error 1003,
  Source Code 69784). Blocked on them; no action available.
- **Third CMT file** — worth confirming with the committee. Their email lists
  *"Camera ready paper, IEEE ecopyright and Abstract Submission in CMT"* — three
  items. The third slot may want the **signed eCopyright form**, not a `.docx`.
  That would explain why it cannot be satisfied yet.
- **Re-upload** the camera-ready to CMT; the copy currently uploaded there is an
  older build without the reviewer response.
- §VII promises *"Code will be released upon acceptance"* — public release still
  pending.
