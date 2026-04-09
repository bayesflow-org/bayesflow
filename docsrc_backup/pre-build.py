import logging
import os
import shutil
import sys
from pathlib import Path

try:
    from sphinx_polyversion.git import Git

    USE_POLYVERSION = True
except ImportError:
    USE_POLYVERSION = False

logging.basicConfig()
logger = logging.getLogger("pre-build.py")
logger.setLevel(logging.DEBUG)


def copy_files(sourcedir):
    basedir = Path(os.path.abspath(sourcedir)).parent.parent
    logger.info(f"Base directory is '{basedir}'")
    logger.info(f"Documentation source directory is '{sourcedir}'")

    # copy examples
    examples_src = os.path.join(basedir, "examples")
    examples_dst = os.path.join(sourcedir, "_examples")
    if os.path.exists(examples_src):
        logger.info("Copying examples")
        shutil.copytree(examples_src, examples_dst, dirs_exist_ok=True)
    examples_in_progress = os.path.join(examples_dst, "in_progress")
    if os.path.exists(examples_in_progress):
        shutil.rmtree(examples_in_progress)
    # copy contributing and installation
    contributing_src = os.path.join(basedir, "CONTRIBUTING.md")
    contributing_dst = os.path.join(sourcedir, "contributing.md")
    if os.path.exists(contributing_src):
        shutil.copy2(contributing_src, contributing_dst)
    installation_src = os.path.join(basedir, "INSTALL.rst")
    installation_dst = os.path.join(sourcedir, "installation.rst")
    if os.path.exists(installation_src):
        shutil.copy2(installation_src, installation_dst)
    # create empty bibtex file if none exists
    bibtex_path = os.path.join(sourcedir, "references.bib")
    if not os.path.exists(bibtex_path):
        open(bibtex_path, "a").close()


def patch_conf(sourcedir):
    if USE_POLYVERSION:
        root = Git.root(Path(__file__).parent)
    else:
        root = str(Path(os.path.abspath(sourcedir)).parent.parent)
    sourcedir = Path(sourcedir)
    cursrc = Path(root) / "docsrc" / "source"
    if os.path.abspath(cursrc) == os.path.abspath(sourcedir):
        return
    # copy the configuration file: shared for all versions
    conf_src = Path(cursrc) / "conf.py"
    conf_dst = Path(sourcedir) / "conf.py"
    if conf_src.exists():
        logger.info("Overwriting old conf.py with current conf.py")
        shutil.copy2(conf_src, conf_dst)
    # copy HTML and CSS for versioning sidebar
    copy_rel_paths = [
        "_static/bayesflow_hor_dark.png",
        "_static/bayesflow_hor.png",
        "_static/custom.css",
        "sphinxext/override_pst_pagetoc.py",
        "sphinxext/adapt_autodoc_docstring.py",
    ]
    for path in copy_rel_paths:
        srcfile = cursrc / path
        dstfile = sourcedir / path
        if srcfile.exists():
            os.makedirs(dstfile.parent, exist_ok=True)
            shutil.copy2(srcfile, dstfile)


if __name__ == "__main__":
    logger.info("Running pre-build script")  # move files around if necessary
    sourcedir = sys.argv[1]
    copy_files(sourcedir)
    patch_conf(sourcedir)
