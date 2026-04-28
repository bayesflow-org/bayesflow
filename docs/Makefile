.PHONY: production-docs-sequential
production-docs-sequential:
	@sphinx-polyversion --sequential poly.py
	@echo 'Copying docs to build/html'
	@rm -rf build/html
	@mkdir -p build/html
	@cp -r _build_polyversion/* build/html
	@cp .nojekyll build/html/.nojekyll
	@rm -rf _build_polyversion

.PHONY: production-docs
production-docs:
	@sphinx-polyversion poly.py
	@echo 'Copying docs to build/html'
	@rm -rf build/html
	@mkdir -p build/html
	@cp -r _build_polyversion/* build/html
	@cp .nojekyll build/html/.nojekyll
	@rm -rf _build_polyversion

.PHONY: local-docs
local-docs:
	@sphinx-polyversion --local poly.py
	@echo 'Copying docs to build/html'
	@rm -rf build/html
	@mkdir -p build/html
	@cp -r _build_polyversion/* build/html
	@cp .nojekyll build/html/.nojekyll
	@rm -rf _build_polyversion

.PHONY: view-docs
view-docs:
	@echo "Serving the contents of 'build/html'... (open the link below to view)."
	@echo "Interrupt with Ctrl+C."
	@python -m http.server -d build/html -b 127.0.0.1 8090

.PHONY: clean
clean:
	@rm -rf build/html
	@rm -rf _build
	@rm -rf $(BUILDDIR)
	@rm -rf _build_polyversion
	@rm -rf source/_examples
	@rm -f source/contributing.md
	@rm -f source/installation.rst

.PHONY: clean-all
clean-all:
	@rm -rf .docs_venvs .bf_doc_gen_venv
	@make clean