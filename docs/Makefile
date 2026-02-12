SOURCEDIR = source
BUILDDIR = build
PORT = 8000

# put this first so that "make" without arguments is the same as "make help"
.PHONY: help Makefile
help:
	@sphinx-build -M help "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: build-docs
build-docs:
	@make html

# Starts a local server and opens the browser to view the docs.
# The server will serve the contents of 'build/html' on http://localhost:8000.
# The browser will automatically open to the correct URL once the server is ready.
.PHONY: serve-docs
serve-docs:
	@echo "\033[36mServing the contents of 'build/html'...\033[0m"
	@echo "\033[36mInterrupt with Ctrl+C.\033[0m"
	@( until curl -s http://localhost:8000 >/dev/null; do sleep 0.2; done; \
	  xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000 ) & \
	python -m http.server 8000 -d build/html >/dev/null 2>&1

.PHONY: run
run:
	@make build-docs
	@make serve-docs

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@sphinx-build -M $@ "$(SOURCEDIR)" "$(BUILDDIR)"
