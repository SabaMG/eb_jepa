# Build the EB-JEPA Hackathon Field Guide.
# The bibliography ships as a precompiled main.bbl (reused from the arXiv source),
# so we do NOT run bibtex; three pdflatex passes resolve refs, citations, cleveref.
#
# Each pass is retried once: this project tree lives on a network filesystem that
# occasionally serves a not-yet-flushed (null-byte) main.aux to pdflatex's
# end-of-run re-read, failing the pass. The failed pass leaves a clean aux on
# disk, so the retry succeeds. The document itself is fine.

MAIN = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex main.bbl
	$(LATEX) $(MAIN).tex || $(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex || $(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex || $(LATEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).blg $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz build.log
