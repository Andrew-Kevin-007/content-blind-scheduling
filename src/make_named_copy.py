"""Build the non-anonymous copy of the manuscript.

The submission PDF must stay anonymous for double-blind review, so this writes a
SEPARATE, obviously-named file rather than touching main.tex. Uploading the
named PDF to CMT would risk a desk reject, hence the filename.

Note: do not attempt this substitution through a bash heredoc. Heredocs on this
machine consume LaTeX backslashes, so the toggle silently fails to flip and you
get a byte-identical anonymous PDF under a misleading name.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
STEM = "main_NAMED_do-not-submit"
TOGGLE_ON = "\\blindreviewtrue"
TOGGLE_OFF = "\\blindreviewfalse"


def main() -> int:
    src = (PAPER / "main.tex").read_text(encoding="utf-8")

    # Replace the DIRECTIVE, not the first textual occurrence. The header
    # comment block mentions \blindreviewtrue several lines earlier, so a naive
    # replace-first edits the comment and silently leaves the toggle set,
    # producing an anonymous PDF under a filename claiming otherwise.
    lines = src.splitlines(keepends=True)
    hits = [
        i for i, ln in enumerate(lines)
        if ln.lstrip().startswith(TOGGLE_ON)
    ]
    if len(hits) != 1:
        print(f"ERROR: expected exactly one {TOGGLE_ON} directive, found {len(hits)}")
        return 1
    lines[hits[0]] = lines[hits[0]].replace(TOGGLE_ON, TOGGLE_OFF, 1)
    print(f"  flipped toggle on line {hits[0] + 1}")

    (PAPER / f"{STEM}.tex").write_text("".join(lines), encoding="utf-8")

    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", f"{STEM}.tex"],
        ["bibtex", STEM],
        ["pdflatex", "-interaction=nonstopmode", f"{STEM}.tex"],
        ["pdflatex", "-interaction=nonstopmode", f"{STEM}.tex"],
    ):
        subprocess.run(cmd, cwd=PAPER, capture_output=True)

    pdf = PAPER / f"{STEM}.pdf"
    if not pdf.exists():
        print("ERROR: build produced no PDF")
        return 1

    txt = subprocess.run(
        ["pdftotext", str(pdf), "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout or ""

    checks = {
        "Kevin Andrew present": "Kevin Andrew" in txt,
        "Kavita Sri present": "Kavita Sri" in txt,
        "both emails present": txt.count("licet.ac.in") >= 2,
        "no longer anonymous": "Anonymous" not in txt,
    }
    for label, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")

    # The submission copy must be untouched and still anonymous.
    sub = subprocess.run(
        ["pdftotext", str(PAPER / "main.pdf"), "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout or ""
    sub_ok = "Anonymous" in sub and "Kevin" not in sub and "licet" not in sub
    print(f"  {'ok  ' if sub_ok else 'FAIL'} main.pdf still anonymous")

    for ext in ("tex", "aux", "log", "bbl", "blg", "out"):
        (PAPER / f"{STEM}.{ext}").unlink(missing_ok=True)

    return 0 if all(checks.values()) and sub_ok else 1


if __name__ == "__main__":
    sys.exit(main())
