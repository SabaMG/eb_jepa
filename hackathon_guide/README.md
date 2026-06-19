# EB-JEPA Hackathon Field Guide

LaTeX source for the participant guide of the 24-hour EB-JEPA hackathon
(100 participants, 25 teams, 72 B200 GPUs). The guide walks teams through the
three `eb_jepa` examples (`image_jepa`, `video_jepa`, `ac_video_jepa`), the
extension points they will edit, and a curated menu of project tracks (flagship:
an EEG-JEPA track on TUH/TUAB inspired by LaBraM).

## Build

```bash
make            # 3 pdflatex passes -> main.pdf
make clean      # remove aux files
```

Requires a TeX Live install with `listings`, `tcolorbox`, `cleveref` (all standard).
The build uses **pdflatex only**: the bibliography is the precompiled `main.bbl`
reused from the EB-JEPA arXiv source (`arxiv_src/`), so there is **no bibtex pass**
and no `.bib` file. If you add citations to new references, add a matching
`\bibitem` to `main.bbl` (or generate a proper `references.bib` and switch the
build to bibtex).

## Layout

```
main.tex                  # the whole guide (single file: preamble + 7 sections)
figures/                  # the 3 figures used, reused from the paper source
main.bbl                  # reused bibliography (72 entries)
iclr2026_conference.sty   # ICLR 2026 conference style (required, not in TeX Live)
iclr2026_conference.bst   # ICLR 2026 bib style (only needed to regenerate the .bbl)
Makefile, README.md, .gitignore
```

Sections, in order within `main.tex`: welcome + 24h game plan; the JEPA recipe in
2 pages (energy view); setup + launcher; the three examples (run/read/tweak);
extension points; the project-track menu (EEG flagship + 10 more); logistics +
B200 notes + judging.

`fancyhdr.sty`/`natbib.sty` are not vendored here: TeX Live provides them.

## Citations

The reusable `\cite` keys live in `main.bbl` (e.g. `IJEPA`, `VJEPA2`, `VICReg`,
`balestriero2025lejepaprovablescalableselfsupervised`, `MPPI`, `TDMPC2`,
`PathAMI`, `sobal2022jepaslowfeatures`, `DINO-WM`, `terver2026JEPAWMs`). Run
`grep -oP '\\bibitem\[[^]]*\]\{\K[^}]+' main.bbl` for the full list.
