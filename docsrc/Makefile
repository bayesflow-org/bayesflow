.PHONY: docs-sequential
docs-sequential:
	@sphinx-polyversion --sequential poly.py
	@echo 'Copying docs to ../docs'
	@rm -rf ../docs/*
	@cp -a _build_polyversion/. ../docs
	@cp .nojekyll ../docs/.nojekyll
	@rm -rf _build_polyversion

.PHONY: docs
docs:
	@sphinx-polyversion poly.py
	@echo 'Copying docs to ../docs'
	@rm -rf ../docs/*
	@cp -a _build_polyversion/. ../docs
	@cp .nojekyll ../docs/.nojekyll
	@rm -rf _build_polyversion

.PHONY: local
local:
	@sphinx-polyversion --local poly.py
	@echo 'Copying docs to ../docs'
	@rm -rf ../docs/*
	@cp -a _build_polyversion/. ../docs
	@cp .nojekyll ../docs/.nojekyll
	@rm -rf _build_polyversion

.PHONY: clean
clean:
	@rm -rf ../docs/*
	@rm -rf _build
	@rm -rf _build_polyversion
	@rm -rf source/_examples
	@rm -f source/installation.rst
	@rm -f source/contributing.md

.PHONY: clean-all
clean-all:
	@make clean
	@rm -rf .docs_venvs .bf_doc_gen_venv
	@find source/api -type f ! -name 'bayesflow.rst' -exec rm -f {} +

.PHONY: view-docs
view-docs:
	@echo "Serving the contents of '../docs'... (open the link below to view)."
	@echo "Interrupt with Ctrl+C."
	@python -m http.server -d ../docs -b 127.0.0.1 8090

.PHONY: help Makefile
