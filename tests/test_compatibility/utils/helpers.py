import pytest
import hashlib
import inspect
from pathlib import Path


class SaveLoadTest:
    filenames = {}

    @pytest.fixture(autouse=True)
    def filepaths(self, data_dir, mode, request):
        # this name contains the config for the test and is therefore a unique identifier
        test_config_str = request._pyfuncitem.name
        # hash it, as it could be too long for the file system
        prefix = hashlib.sha1(test_config_str.encode("utf-8")).hexdigest()
        # use path to test file as base, remove ".py" suffix
        base_path = Path(inspect.getsourcefile(type(self))[:-3])
        # add class name
        directory = base_path / type(self).__name__
        # only keep the path relative to the tests directory
        directory = directory.relative_to(Path("tests").absolute())
        directory = Path(data_dir) / directory

        if mode == "save":
            directory.mkdir(parents=True, exist_ok=True)

        files = {}
        for label, filename in self.filenames.items():
            path = directory / f"{prefix}__{filename}"
            if mode == "load" and not path.exists():
                pytest.skip(f"Required file not available: {path}")
            files[label] = path
        return files
