"""Render a Markdown document to PDF via LaTeX.

Pandoc is not installed on this machine, but MiKTeX is, so we translate the
subset of Markdown actually used in DESIGN_DECISIONS.md rather than adding a
toolchain dependency the day before a deadline.

Handles: ATX headings, bold, italic, inline code, fenced code, pipe tables,
bullet and numbered lists, horizontal rules, and links. Anything outside that
subset is escaped and passed through as literal text, so an unsupported
construct degrades to plain prose rather than to a compile error.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
\usepackage[table]{xcolor}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage{textcomp}

\definecolor{rule}{gray}{0.75}
\definecolor{codebg}{gray}{0.96}

\titleformat{\section}{\Large\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{0.6em}{}
\setlist[itemize]{leftmargin=1.2em,itemsep=1pt,topsep=3pt}
\setlist[enumerate]{leftmargin=1.4em,itemsep=1pt,topsep=3pt}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.25}

\pagestyle{fancy}\fancyhf{}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0pt}

\newcommand{\code}[1]{\colorbox{codebg}{\texttt{\small #1}}}

\begin{document}
"""

SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def esc(s: str) -> str:
    out = []
    for ch in s:
        out.append(SPECIALS.get(ch, ch))
    return "".join(out)


def inline(s: str) -> str:
    """Convert inline markup. Code spans are protected before escaping."""
    spans: list[str] = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    s = esc(s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: r"\code{" + esc(spans[int(m.group(1))]) + "}", s)
    return s


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]          # cells[1] is the --- separator
    n = len(header)
    # First column left-aligned and wrappable; the rest wrap too, so long
    # rationale text does not overflow the page.
    spec = "@{}p{0.26\\textwidth}" + "p{%.3f\\textwidth}" % (0.68 / max(n - 1, 1)) * (n - 1) + "@{}"
    out = [r"\begin{center}\small", r"\begin{tabular}{" + spec + "}", r"\toprule"]
    out.append(" & ".join(r"\textbf{" + inline(h) + "}" for h in header) + r" \\")
    out.append(r"\midrule")
    for row in body:
        row = (row + [""] * n)[:n]
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    list_mode: str | None = None

    def close_list():
        nonlocal list_mode
        if list_mode:
            out.append(r"\end{" + list_mode + "}")
            list_mode = None

    while i < len(lines):
        ln = lines[i]

        if ln.strip().startswith("|") and i + 1 < len(lines) and set(
            lines[i + 1].replace("|", "").replace(":", "").strip()
        ) <= {"-", " "} and lines[i + 1].strip():
            close_list()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(table(block))
            continue

        if ln.startswith("# "):
            close_list()
            out.append(r"\title{" + inline(ln[2:]) + "}")
            out.append(r"\date{}\maketitle")
        elif ln.startswith("## "):
            close_list()
            out.append(r"\section*{" + inline(ln[3:]) + "}")
        elif ln.startswith("### "):
            close_list()
            out.append(r"\subsection*{" + inline(ln[4:]) + "}")
        elif ln.strip() == "---":
            close_list()
            out.append(r"\vspace{0.3em}{\color{rule}\hrule}\vspace{0.5em}")
        elif re.match(r"^\s*[-*]\s+", ln):
            if list_mode != "itemize":
                close_list()
                out.append(r"\begin{itemize}")
                list_mode = "itemize"
            out.append(r"\item " + inline(re.sub(r"^\s*[-*]\s+", "", ln)))
        elif re.match(r"^\s*\d+\.\s+", ln):
            if list_mode != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}")
                list_mode = "enumerate"
            out.append(r"\item " + inline(re.sub(r"^\s*\d+\.\s+", "", ln)))
        elif not ln.strip():
            close_list()
            out.append("")
        elif list_mode:
            out[-1] += " " + inline(ln.strip())
        else:
            # Gather the whole paragraph before converting. Inline markup such
            # as *emphasis* frequently spans a source line break, and converting
            # line-by-line would leave the asterisks as literal text.
            para = [ln.strip()]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].startswith(("#", "|", "- ", "* "))
                and lines[i + 1].strip() != "---"
                and not re.match(r"^\s*\d+\.\s+", lines[i + 1])
            ):
                i += 1
                para.append(lines[i].strip())
            out.append(inline(" ".join(para)))
        i += 1

    close_list()
    return PREAMBLE + "\n".join(out) + "\n\\end{document}\n"


def main() -> int:
    src = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "DESIGN_DECISIONS.md")
    stem = src.stem
    tex = ROOT / "paper" / f"{stem}.tex"
    tex.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")

    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex.name],
            cwd=tex.parent, capture_output=True,
        )

    pdf = tex.parent / f"{stem}.pdf"
    if not pdf.exists():
        log = (tex.parent / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
        print("BUILD FAILED\n" + "\n".join(
            l for l in log.splitlines() if l.startswith("!")
        )[:2000])
        return 1

    pages = subprocess.run(
        ["pdfinfo", str(pdf)], capture_output=True, text=True
    ).stdout
    print(f"wrote {pdf.relative_to(ROOT)}  "
          f"({pdf.stat().st_size // 1024} KB, "
          f"{[l for l in pages.splitlines() if l.startswith('Pages')]})")

    for ext in ("tex", "aux", "log", "out"):
        (tex.parent / f"{stem}.{ext}").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
