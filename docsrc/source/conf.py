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
    data = load(globals())
    current = data["current"].name
except ImportError:
    USE_POLYVERSION = False
    print("sphinx_polyversion not installed, building single version")
    current = "local"

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("sphinxext"))

# might set copyright end to wrong year -> remove
if "SOURCE_DATE_EPOCH" in os.environ:
    del os.environ["SOURCE_DATE_EPOCH"]

# -- Project information -----------------------------------------------------

project = "BayesFlow"
author = "The BayesFlow authors"
copyright = "2023-%Y, BayesFlow authors (lead maintainer: Stefan T. Radev)"


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

if not current.startswith("v1."):
    extensions.extend(
        [
            "override_pst_pagetoc",  # local, see sphinxext folder
            "adapt_autodoc_docstring",  # local, see sphinxext folder
        ]
    )

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

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

# Define shorthand for external links:
extlinks = {
    "mainbranch": (f"https://github.com/bayesflow-org/bayesflow/blob/{current}/%s", None),
}

coverage_show_missing_items = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Options for autodoc and autosummary
# do not ignore __all__, use it to determine public members
autosummary_ignore_module_all = False
autosummary_imported_members = False
# selects content to insert into the main body of an autoclass directive.
autoclass_content = "both"

if current.startswith("v1."):
    autodoc_default_options = {
        "members": True,
        "undoc-members": True,
        "imported-members": False,
        "inherited-members": True,
        "show-inheritance": True,
        "special-members": "__call__",
        "memberorder": "bysource",
    }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_title = "BayesFlow: Amortized Bayesian Inference"

# Add any paths that contain custom _static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin _static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "use_edit_page_button": True,
    "logo": {
        "alt-text": "BayesFlow",
        "image_light": "_static/bayesflow_hor.png",
        "image_dark": "_static/bayesflow_hor_dark.png",
    },
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bayesflow-org/bayesflow",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Discourse Forum",
            "url": "https://discuss.bayesflow.org/",
            "icon": "fa-brands fa-discourse",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "left",
    # -- Template placement in theme layouts ----------------------------------
    "navbar_start": ["navbar-logo"],
    # Note that the alignment of navbar_center is controlled by navbar_align
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    # navbar_persistent is persistent right (even when on mobiles)
    "navbar_persistent": ["search-button"],
    "switcher": {
        "json_url": "/versions.json",
        "version_match": current,
    },
    "check_switcher": False,
}
html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise site
    "github_user": "bayesflow-org",
    "github_repo": "bayesflow",
    "github_version": current,
    "doc_path": "docsrc/source",
}
html_logo = "_static/bayesflow_hor.png"
html_favicon = "_static/bayesflow_hex.ico"
html_baseurl = "https://www.bayesflow.org/"

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
