# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

try:
    from sphinx_polyversion import load
    from sphinx_polyversion.git import GitRef
    from sphinx_polyversion.api import LoadError

    USE_POLYVERSION = True
except ImportError:
    USE_POLYVERSION = False
    print("sphinx_polyversion not installed, building single version")

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "BayesFlow"
author = "The BayesFlow authors"
copyright = "2023, BayesFlow authors (lead maintainer: Stefan T. Radev)"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["references.bib"]

numpydoc_show_class_members = False
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ["http", "https", "mailto"]
autodoc_default_options = {
    "members": "var1, var2",
    "special-members": "__call__,__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "member-order": "bysource",
}

# Define shorthand for external links:
extlinks = {
    "mainbranch": ("https://github.com/bayesflow-org/bayesflow/blob/master/%s", None),
}

coverage_show_missing_items = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
autosummary_imported_members = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = "BayesFlow: Amortized Bayesian Inference"

# Add any paths that contain custom _static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin _static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "repository_url": "https://github.com/bayesflow-org/bayesflow",
    "repository_branch": "master",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "logo": {"alt-text": "BayesFlow"},
}
html_logo = "_static/bayesflow_hex.png"
html_favicon = "_static/bayesflow_hex.ico"
html_baseurl = "https://www.bayesflow.org/"
html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"]
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
        "versioning.html",
    ],
}

todo_include_todos = True

# do not execute jupyter notebooks when building docs
nb_execution_mode = "off"

# download notebooks as .ipynb and not as .ipynb.txt
html_sourcelink_suffix = ""

suppress_warnings = [
    f"autosectionlabel._examples/{filename.split('.')[0]}"
    for filename in os.listdir("../../examples")
    if os.path.isfile(os.path.join("../../examples", filename))
]  # Avoid duplicate label warnings for Jupyter notebooks.

remove_from_toctrees = ["_autosummary/*"]

autosummmary_generate = True

# versioning data for template
if USE_POLYVERSION:
    try:
        data = load(globals())
        current: GitRef = data["current"]
    except LoadError:
        print("sphinx_polyversion could not load. Building single version")
