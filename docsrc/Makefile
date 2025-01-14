# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# from https://github.com/acerbilab/pyvbmc/blob/main/docsrc/Makefile
# copy example notebooks to source folder
# render docs
# copy rendered docs and .nojekyll (for github.io) to docs folder
# remove example notebooks from source folder
dev:
	@python pre-build.py source
	@make html
	@echo 'Copying docs to ../docs'
	@cp -a _build/html/. ../docs
	@cp .nojekyll ../docs/.nojekyll

github:
	@BF_DOCS_SEQUENTIAL_BUILDS=1 sphinx-polyversion poly.py
	@echo 'Copying docs to ../docs'
	@cp -a _build_polyversion/. ../docs
	@cp .nojekyll ../docs/.nojekyll

parallel:
	@BF_DOCS_SEQUENTIAL_BUILDS=0 sphinx-polyversion poly.py
	@echo 'Copying docs to ../docs'
	@cp -a _build_polyversion/. ../docs
	@cp .nojekyll ../docs/.nojekyll

clean:
	@rm -rf ../docs/*
	@rm -rf _build
	@rm -rf _build_polyversion
	@rm -rf source/_examples
	@rm -f source/installation.rst
	@rm -f source/contributing.md

clean-all:
	@make clean
	@rm -rf docsrc/.bf_doc_gen_venv
	@find source/api -type f ! -name 'bayesflow.rst' -exec rm -f {} +

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
