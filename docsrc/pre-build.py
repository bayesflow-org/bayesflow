import os
import shutil
import sys
from pathlib import Path

try:
    from sphinx_polyversion.git import Git

    USE_POLYVERSION = True
except ImportError:
    USE_POLYVERSION = False


def copy_files(sourcedir):
    basedir = Path(os.path.abspath(sourcedir)).parent.parent
    print(basedir, sourcedir)

    # copy examples
    examples_src = os.path.join(basedir, "examples")
    examples_dst = os.path.join(sourcedir, "_examples")
    if os.path.exists(examples_src):
        print("Copying examples")
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
    cursrc = os.path.join(root, "docsrc", "source")
    if os.path.abspath(cursrc) == os.path.abspath(sourcedir):
        return
    # copy the configuration file: shared for all versions
    conf_src = os.path.join(cursrc, "conf.py")
    conf_dst = os.path.join(sourcedir, "conf.py")
    if os.path.exists(conf_src):
        print("Overwriting old conf.py with current conf.py")
        shutil.copy2(conf_src, conf_dst)
    # copy HTML and CSS for versioning sidebar
    versioning_src = os.path.join(cursrc, "_templates", "versioning.html")
    versioning_dst = os.path.join(sourcedir, "_templates", "versioning.html")
    if os.path.exists(versioning_src):
        os.makedirs(os.path.join(sourcedir, "_templates"), exist_ok=True)
        shutil.copy2(versioning_src, versioning_dst)
    css_src = os.path.join(cursrc, "_static", "custom.css")
    css_dst = os.path.join(sourcedir, "_static", "custom.css")
    if os.path.exists(css_src):
        os.makedirs(os.path.join(sourcedir, "_static"), exist_ok=True)
        shutil.copy2(css_src, css_dst)


if __name__ == "__main__":
    print("Running pre-build script")  # move files around if necessary
    sourcedir = sys.argv[1]
    copy_files(sourcedir)
    patch_conf(sourcedir)
