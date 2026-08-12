PYTHON ?= python3
PANDOC ?= pandoc
PDF_ENGINE ?= xelatex
.PHONY: all figures pdf clean check

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

clean:
	rm -f figures/*.png paper.pdf
