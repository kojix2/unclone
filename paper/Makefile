PYTHON ?= python3
PANDOC ?= pandoc
PDF_ENGINE ?= xelatex
JA_PDF_ENGINE ?= lualatex
.PHONY: all figures pdf ja clean check

all: pdf

check:
	@$(PYTHON) -c 'import matplotlib'
	@$(PANDOC) --version >/dev/null
	@$(PDF_ENGINE) --version >/dev/null

figures:
	$(PYTHON) scripts/plot_figures.py

pdf: figures
	$(PANDOC) paper.md \
		--from=markdown \
		--citeproc \
		--pdf-engine=$(PDF_ENGINE) \
		--output=paper.pdf

ja: figures
	$(PANDOC) paper-ja.md \
		--from=markdown \
		--citeproc \
		--pdf-engine=$(JA_PDF_ENGINE) \
		--output=paper-ja.pdf

clean:
	rm -f figures/*.png paper.pdf paper-ja.pdf
