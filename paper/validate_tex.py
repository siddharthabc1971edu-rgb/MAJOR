"""Static sanity checks for the IEEE LaTeX manuscript (no TeX toolchain needed)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "ieee_paper.tex"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    # Braces, ignoring escaped \{ and \}.
    opens = len(re.findall(r"(?<!\\)\{", text))
    closes = len(re.findall(r"(?<!\\)\}", text))
    print(f"Braces: {opens} open / {closes} close ->", "OK" if opens == closes else "MISMATCH")

    # Environments.
    begins = Counter(re.findall(r"\\begin\{([^}]+)\}", text))
    ends = Counter(re.findall(r"\\end\{([^}]+)\}", text))
    print("Environments:", "OK" if begins == ends else "MISMATCH")
    for env in sorted(set(begins) | set(ends)):
        if begins[env] != ends[env]:
            print(f"  !! {env}: {begins[env]} begin / {ends[env]} end")

    # Citations and bibliography.
    cites: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        cites.update(key.strip() for key in group.split(","))
    bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    print(f"Citations: {len(cites)} / Bibitems: {len(bibs)}")
    for key in sorted(cites - bibs):
        print(f"  !! missing bibitem: {key}")
    for key in sorted(bibs - cites):
        print(f"  !! unused bibitem: {key}")

    # Labels and references.
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
    print(f"Labels: {len(labels)} / Refs: {len(refs)}")
    for key in sorted(refs - labels):
        print(f"  !! undefined reference: {key}")
    for key in sorted(labels - refs):
        print(f"  !! unused label: {key}")

    # Figures.
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    for fig in figures:
        path = PAPER / fig
        print(f"Figure {fig}:", "OK" if path.exists() else "MISSING")

    # Tabular column consistency (accounting for \multicolumn spans).
    tables = re.findall(r"\\begin\{tabular\}\{((?:@\{[^}]*\}|[^}])*)\}(.*?)\\end\{tabular\}",
                        text, re.DOTALL)
    print(f"Tabulars: {len(tables)}")
    for number, (spec, body) in enumerate(tables, 1):
        rows = [row for row in re.split(r"\\\\", body) if row.strip() and "\\toprule" not in row
                and "\\midrule" not in row and "\\bottomrule" not in row and "\\cmidrule" not in row]
        widths: set[int] = set()
        for row in rows:
            extra = sum(int(m) - 1 for m in re.findall(r"\\multicolumn\{(\d+)\}", row))
            widths.add(row.replace(r"\&", "").count("&") + 1 + extra)
        spec_width = len(re.findall(r"[lcr]", re.sub(r"@\{[^}]*\}", "", spec)))
        print(f"  table {number}: spec width {spec_width} / row widths {sorted(widths)}",
              "OK" if len(widths) == 1 and spec_width in widths else "!! CHECK")
        if len(widths) != 1 or spec_width not in widths:
            for row in rows:
                print(f"      amps={row.count('&')}: {row.strip()[:90]}")

    # Non-ASCII characters (should be none; everything uses LaTeX macros).
    non_ascii = sorted({ch for ch in text if ord(ch) > 127})
    print("Non-ASCII characters:", non_ascii if non_ascii else "none")
    print("Done.")


if __name__ == "__main__":
    main()
