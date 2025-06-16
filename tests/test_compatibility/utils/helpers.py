import pytest
from utils import get_valid_filename, get_path


class SaveLoadTest:
    filenames = {}

    @pytest.fixture(autouse=True)
    def filepaths(self, data_dir, mode, request):
        prefix = get_valid_filename(request._pyfuncitem.name)
        files = {}
        for label, filename in self.filenames.items():
            path = get_path(data_dir, f"{prefix}__{filename}", create=mode == "save")
            if mode == "load" and not path.exists():
                pytest.skip(f"Required file not available: {path}")
            files[label] = path
        return files
