"""
Build compact and full Gitingest LLM-context bundles.

On release, generates:

- llm_context/llm_context_compact_<tag>.md
- llm_context/llm_context_full_<tag>.md

Old context files in ``llm_context/`` are removed before writing new ones.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

# --- Paths and config ---
BASE_DIR = Path(__file__).parent.parent.resolve()
README_FILE = BASE_DIR / "README.md"
EXAMPLES_DIR = BASE_DIR / "examples"
SRC_DIR = BASE_DIR / "bayesflow"
OUTPUT_DIR = BASE_DIR / "llm_context"

EXCLUDED_DIR_NAMES = ["experimental"]


def convert_notebooks_to_md(src_dir: Path, dst_dir: Path) -> List[Path]:
    """Convert Jupyter notebooks (*.ipynb) to Markdown."""
    created: List[Path] = []
    for ipynb_file in sorted(src_dir.glob("*.ipynb")):
        notebook = json.loads(ipynb_file.read_text(encoding="utf-8"))
        parts: List[str] = []
        for cell in notebook.get("cells", []):
            src = "".join(cell.get("source", []))
            if cell.get("cell_type") == "markdown":
                parts.append(src)
            elif cell.get("cell_type") == "code":
                parts.append(f"```python\n{src}\n```")
        if parts:
            md_file = dst_dir / f"{ipynb_file.stem}.md"
            md_file.write_text("\n\n".join(parts), encoding="utf-8")
            created.append(md_file.resolve())
    if not created:
        raise FileNotFoundError("No example notebooks (*.ipynb) found.")
    return created


def collect_py_files(root: Path, exclude: Sequence[str] = ()) -> List[Path]:
    """Collect Python source files from a directory."""
    excluded = set(exclude)
    return sorted(
        f.resolve()
        for f in root.rglob("*.py")
        if not any(p.name in excluded for p in f.parents)
    )


def run_gitingest(work_dir: Path, output: Path, exclude: Sequence[str] | None = None) -> None:
    """Run gitingest on a directory."""
    cmd = ["gitingest", str(work_dir), "--output", str(output)]
    if exclude:
        for pat in exclude:
            cmd.extend(["--exclude-pattern", pat])
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.stderr.write("ERROR: 'gitingest' not found. Install and add to PATH.\n")
        raise
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"ERROR: gitingest failed (exit code {e.returncode}).\n")
        raise
    print(f"Gitingest executed; output saved to {output}")


def main() -> None:
    """Build compact and full LLM context bundles with versioned filenames."""
    tag = (sys.argv[1] if len(sys.argv) > 1 else None) or "dev"

    if not README_FILE.exists():
        raise FileNotFoundError(f"Missing README.md: {README_FILE}")
    if not EXAMPLES_DIR.exists():
        raise FileNotFoundError(f"Missing examples dir: {EXAMPLES_DIR}")

    # Clean old context files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT_DIR.glob("llm_context_*.md"):
        old.unlink()

    compact_output = OUTPUT_DIR / f"llm_context_compact_{tag}.md"
    full_output = OUTPUT_DIR / f"llm_context_full_{tag}.md"

    with (
        tempfile.TemporaryDirectory(prefix="examples_", dir=BASE_DIR) as tmp_examples,
        tempfile.TemporaryDirectory(prefix="compact_", dir=BASE_DIR) as tmp_compact,
        tempfile.TemporaryDirectory(prefix="full_", dir=BASE_DIR) as tmp_full,
    ):
        tmp_examples = Path(tmp_examples)
        tmp_compact = Path(tmp_compact)
        tmp_full = Path(tmp_full)

        # Convert notebooks
        example_mds = convert_notebooks_to_md(EXAMPLES_DIR, tmp_examples)

        # ==== Compact bundle ====
        (tmp_compact / "examples").mkdir(parents=True, exist_ok=True)
        shutil.copy(README_FILE, tmp_compact / "README.md")
        for md in example_mds:
            shutil.copy(md, tmp_compact / "examples" / md.name)
        run_gitingest(tmp_compact, compact_output)

        # ==== Full bundle ====
        (tmp_full / "examples").mkdir(parents=True, exist_ok=True)
        shutil.copy(README_FILE, tmp_full / "README.md")
        for md in example_mds:
            shutil.copy(md, tmp_full / "examples" / md.name)

        if SRC_DIR.exists():
            for pyfile in collect_py_files(SRC_DIR, EXCLUDED_DIR_NAMES):
                rel = pyfile.relative_to(SRC_DIR)
                dest = tmp_full / "bayesflow" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(pyfile, dest)
        else:
            sys.stderr.write(f"WARNING: source dir not found: {SRC_DIR}\n")

        exclude = [f"**/{d}/**" for d in EXCLUDED_DIR_NAMES] if EXCLUDED_DIR_NAMES else None
        run_gitingest(tmp_full, full_output, exclude)


if __name__ == "__main__":
    main()
