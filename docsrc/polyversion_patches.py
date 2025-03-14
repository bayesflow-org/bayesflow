import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
from subprocess import PIPE, CalledProcessError

from sphinx_polyversion.builder import BuildError
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.pyvenv import Pip
from sphinx_polyversion.json import JSONable

from typing import Iterator

import tempfile

logging.basicConfig()
logger = logging.getLogger("poly.py")


# adapted from Pip
class DynamicPip(Pip):
    def __init__(self, path: Path, name: str, venv: str | Path, *args, **kwargs):
        """
        Adapt `Pip` to dynamically use name in venv path.

        See `Pip` for detailed docs.
        """
        logger.info("Setting dynamic venv name: " + str(Path(venv) / name))
        super().__init__(path, name, Path(venv) / name, *args, **kwargs)

    async def __aenter__(self):
        """
        Set the venv up.

        Raises
        ------
        BuildError
            Running `pip install` failed.

        """
        await super(Pip, self).__aenter__()

        logger.info("Running `pip install`...")

        cmd: list[str] = ["pip", "install"]
        cmd += self.args

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.path,
            env=self.activate(self.apply_overrides(os.environ.copy())),
            stdout=PIPE,
            stderr=PIPE,
        )
        out, err = await process.communicate()
        out = out.decode(errors="ignore")
        err = err.decode(errors="ignore")

        self.logger.debug("Installation output:\n %s", out)
        if process.returncode != 0:
            # see https://github.com/real-yfprojects/sphinx-polyversion/pull/32
            self.logger.error("Installation error:\n %s", err)
            raise BuildError from CalledProcessError(
                returncode=process.returncode, cmd=" ".join(cmd), output=out, stderr=err
            )
        return self


class PyDataVersionEncoder(json.JSONEncoder):
    """Encoder to turn list of built GitRefs into a version.json consumable by Sphinx."""

    def transform(self, o: JSONable):
        output = []
        processed_names = []
        for ref in o:
            if ref.name in processed_names:
                continue
            output.append(
                {
                    "name": ref.name,
                    "version": ref.name,
                    "url": f"/{ref.name}",
                }
            )
            processed_names.append(ref.name)
        # do not use cast for performance reasons
        return output

    def __call__(self, o: JSONable):
        return self.transform(o)

    def iterencode(self, o: JSONable, _one_shot: bool = False) -> Iterator[str]:
        """Encode an object."""
        # called for every top level object to encode
        return super().iterencode(self.transform(o), _one_shot)


class CustomDriver(DefaultDriver):
    """DefaultDriver which does not copy venvs for local builds.

    See https://github.com/real-yfprojects/sphinx-polyversion/pull/31
    """

    async def build_local(self) -> None:
        if not self.mock:
            raise ValueError("Missing mock data.")

        # process mock data
        self.targets = self.builds = self.mock["revisions"]
        rev = self.mock["current"]

        if rev not in self.targets:
            self.targets.append(rev)

        # create builder
        builder = await self.init_builder(rev)

        async def get_unignored_files(directory: Path) -> list[Path]:
            cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
            process = await asyncio.create_subprocess_exec(*cmd, cwd=directory, stdout=PIPE)
            out, err = await process.communicate()
            files = out.decode().split("\n")
            return [Path(path.strip()) for path in files]

        # create temporary directory to use for building this version
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            # copy source files
            logger.info(f"Copying source files to temporary directory (except for files ignored by git)... '{tmp}'")
            try:
                files = await get_unignored_files(self.root)
                for filename in files:
                    source = self.root / filename
                    target = path / filename
                    if source.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            shutil.copy2(source, target, follow_symlinks=False)
            except CalledProcessError:
                logger.warning("Could not list un-ignored files using git. Copying full working directory...")
                shutil.copytree(self.root, path, symlinks=True, dirs_exist_ok=True)
            # setup build environment (e.g. poetry/pip venv)
            async with await self.init_environment(path, rev) as env:
                # construct metadata to pass to the build process
                data = await self.init_data(rev, env)
                # build the docs
                artifact = await builder.build(env, self.output_dir / "local", data=data)

        # call hook for success, on failure an exception will have been raised
        self.build_succeeded(rev, artifact)

        await self.build_root()
