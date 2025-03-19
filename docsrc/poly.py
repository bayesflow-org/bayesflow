from datetime import datetime
import logging
from pathlib import Path
from docsrc.polyversion_patches import DynamicPip, CustomDriver, PyDataVersionEncoder

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate, refs_by_type
from sphinx_polyversion.pyvenv import Environment, VenvWrapper
from sphinx_polyversion.sphinx import SphinxBuilder, Placeholder


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#: Determine repository root directory
root = Git.root(Path(__file__).parent)

#: CodeRegex matching the branches to build docs for
BRANCH_REGEX = r"^(main|stable-legacy)$"

#: Regex matching the tags to build docs for
# TAG_REGEX = r"^v((?!1\.)[\.0-9]*)$"
TAG_REGEX = r""

#: Output dir relative to project root
OUTPUT_DIR = "_build_polyversion"

#: Source directory
SOURCE_DIR = "docsrc"

MOCK_DATA = {
    "revisions": [],
    "current": GitRef("local", "", "", GitRefType.BRANCH, datetime.now()),
}

#: Whether to build using only local files and mock data
MOCK = False

#: Whether to run the builds in sequence or in parallel
SEQUENTIAL = False

#: Arguments to pass to `sphinx-build`
SPHINX_ARGS = "-a -v"

#: Extra packages for building docs
SPHINX_DEPS = [
    "sphinx",
    "numpydoc",
    "myst-nb",
    "sphinx_design",
    "pydata-sphinx-theme",
    "sphinxcontrib-bibtex",
    "sphinx-polyversion==1.1.0",
]

#: Extra dependencies to iinstall for version 1
V1_BACKEND_DEPS = [
    "tf-keras",
]

#: Extra dependencies to install for version 2
V2_BACKEND_DEPS = [
    "jax",
    # "torch",
    # "tensorflow",
]

VENV_DIR_NAME = ".docs_venvs"


#: Data passed to templates
def data(driver, rev, env):
    revisions = driver.targets
    branches, tags = refs_by_type(revisions)
    latest = max(tags or branches)
    for b in branches:
        if b.name == "main":
            latest = b
            break

    # sort tags and branches by date, newest first
    return {
        "current": rev,
        "tags": sorted(tags, reverse=True),
        "branches": sorted(branches, reverse=True),
        "revisions": revisions,
        "latest": latest,
    }


def root_data(driver):
    revisions = driver.builds
    branches, tags = refs_by_type(revisions)
    latest = max(tags or branches)
    for b in branches:
        if b.name == "main":
            latest = b
            break
    return {"revisions": revisions, "latest": latest}


# Load overrides read from commandline to global scope
apply_overrides(globals())


# Setup environments for the different versions
src = Path(SOURCE_DIR)
vcs = Git(
    branch_regex=BRANCH_REGEX,
    tag_regex=TAG_REGEX,
    buffer_size=1 * 10**9,  # 1 GB
    predicate=file_predicate([src]),  # exclude refs without source dir
)


creator = VenvWrapper(with_pip=True)


async def selector(rev, keys):
    """Select configuration based on revision"""
    # map all v1 revisions to one configuration
    if rev.name.startswith("v1.") or rev.name == "stable-legacy":
        return "stable-legacy"
    elif rev.name == "local":
        return "local"
    # common config for everything else
    return None


shared_env_kwargs = dict(temporary=SEQUENTIAL, creator=creator, venv=Path(VENV_DIR_NAME))
ENVIRONMENT = {
    # configuration for v2 and dev
    None: DynamicPip.factory(
        **shared_env_kwargs,
        args=["-e", "."] + SPHINX_DEPS + V2_BACKEND_DEPS,
        env={"KERAS_BACKEND": "jax"},
    ),
    # configuration for v1
    "stable-legacy": DynamicPip.factory(
        **shared_env_kwargs,
        args=["-e", "."] + SPHINX_DEPS + V1_BACKEND_DEPS,
        env={"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BAYESFLOW": "1.1.6"},
    ),
    # use local envorinment for local build
    "local": Environment.factory(),
}


# Setup driver and run it
CustomDriver(
    root,
    OUTPUT_DIR,
    vcs=vcs,
    builder=SphinxBuilder(
        src / "source",
        args=SPHINX_ARGS.split(),
        pre_cmd=["python", root / src / "pre-build.py", Placeholder.SOURCE_DIR],
    ),
    env=ENVIRONMENT,
    encoder=PyDataVersionEncoder(),
    selector=selector,
    template_dir=root / src / "polyversion/templates",
    static_dir=root / src / "polyversion/static",
    data_factory=data,
    root_data_factory=root_data,
    mock=MOCK_DATA,
).run(MOCK, SEQUENTIAL)
