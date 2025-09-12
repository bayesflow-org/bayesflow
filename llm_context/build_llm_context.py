#!/usr/bin/env python3
"""
Build two Repomix LLM-context files:

- llm_context/llm_context_compact.md  -> README + examples only
- llm_context/llm_context_full.md     -> README + examples + bayesflow source code

Example notebooks (.ipynb) are converted to temporary Markdown files for clean Repomix conversion.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

# --- Paths and config ---
base_dir = Path(__file__).parent.parent.resolve()
readme_file = base_dir / "README.md"
examples_dir = base_dir / "examples"
src_dir = base_dir / "bayesflow"
output_dir = base_dir / "llm_context"
compact_output_file = output_dir / "llm_context_compact.md"
full_output_file = output_dir / "llm_context_full.md"

EXCLUDED_DIR_NAMES = ["experimental"]

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Safety checks
if not examples_dir.exists():
    print(f"ERROR: examples directory not found: {examples_dir}", file=sys.stderr)
    raise SystemExit(1)
if not src_dir.exists():
    print(f"WARNING: bayesflow source directory not found: {src_dir} -- full context will be skipped.", file=sys.stderr)


def convert_notebooks_to_md_in_temp(src_examples_dir: Path, temp_examples_dir: Path) -> List[Path]:
    """
    Convert Jupyter notebooks (*.ipynb) in a source directory to Markdown files.

    Notes are saved into a temporary examples directory, leaving the original examples/
    untouched. Markdown files are created with code cells fenced as Python blocks.

    Parameters
    ----------
    src_examples_dir : Path
        Directory containing the source *.ipynb notebooks (non-recursive).
    temp_examples_dir : Path
        Temporary directory where the generated *.md files will be written.

    Returns
    -------
    List[Path]
        Absolute paths to the created Markdown files.

    Raises
    ------
    SystemExit
        If no notebooks are found or conversion yields no Markdown content.
    """
    created_md_paths: List[Path] = []

    for ipynb_file in sorted(src_examples_dir.glob("*.ipynb")):
        with ipynb_file.open("r", encoding="utf-8") as f:
            notebook = json.load(f)

        parts: List[str] = []
        for cell in notebook.get("cells", []):
            ctype = cell.get("cell_type")
            src = "".join(cell.get("source", []))
            if ctype == "markdown":
                parts.append(src)
            elif ctype == "code":
                parts.append(f"```python\n{src}\n```")

        # Skip empty conversions (e.g., empty notebook)
        if not parts:
            continue

        md_file = temp_examples_dir / f"{ipynb_file.stem}.md"

        with md_file.open("w", encoding="utf-8") as f:
            f.write("\n\n".join(parts))

        created_md_paths.append(md_file.resolve())

    if not created_md_paths:
        raise FileNotFoundError("No example notebooks (*.ipynb) found or conversion produced no markdown files.")

    return created_md_paths


def collect_py_abs_paths(dir: Path, excluded_dir_names: Sequence[str] = EXCLUDED_DIR_NAMES) -> List[Path]:
    """
    Collect absolute paths to Python files under a directory, excluding certain folder names.

    Parameters
    ----------
    dir : Path
        Root directory to scan for *.py files (recursive).
    excluded_dir_names : Sequence[str], optional
        Directory names to exclude at any depth, e.g., experimental folders.

    Returns
    -------
    List[Path]
        Sorted list of absolute paths to included Python files.
    """
    excluded = set(excluded_dir_names)
    return sorted(
        p.resolve()
        for p in dir.rglob("*.py")
        if not any(parent.name in excluded for parent in p.parents)
    )


def run_repomix_with_file_list(
    file_paths: Sequence[Path],
    output_path: Path,
    repo_cwd: Path,
) -> None:
    """
    Run Repomix to bundle a list of files into a single Markdown output.

    Parameters
    ----------
    file_paths : Sequence[Path]
        Files to include in the Repomix run. Paths may be absolute or relative to repo_cwd.
    output_path : Path
        Destination for the generated Markdown output.
    repo_cwd : Path
        Repository root to use as the working directory for Repomix.

        Raises
    ------
    ValueError
        If file_paths is empty.
    FileNotFoundError
        If the 'repomix' executable is not found on PATH.
    RuntimeError
        If the Repomix command fails.
    """
    if not file_paths:
        raise ValueError(f"No files provided for repomix output: {output_path}")

    cmd = [
        "repomix",
        "--style",
        "markdown",
        "--stdin",
        "-o",
        str(output_path),
    ]

    # Prepare file path list
    stdin_input = "\n".join(str(p) for p in file_paths) + "\n"

    try:
        subprocess.run(cmd, input=stdin_input, text=True, check=True, cwd=str(repo_cwd))
    except FileNotFoundError as e:
        raise FileNotFoundError("'repomix' not found on PATH. Please install it and retry.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Repomix failed with exit code {e.returncode}.") from e

    print(f"Repomix packaged output saved to {output_path}")


def to_relative_paths(paths: Iterable[Path], base: Path) -> List[Path]:
    """
    Convert a list of paths to paths relative to a base directory when possible.

    Parameters
    ----------
    paths : Iterable[Path]
        Paths to convert.
    base : Path
        Base directory.

    Returns
    -------
    List[Path]
        Relative paths if conversion succeeds; otherwise original paths.
    """
    rels: List[Path] = []
    for p in paths:
        try:
            rels.append(p.relative_to(base))
        except Exception:
            rels.append(p)
    return rels


def main() -> None:
    """
    Entry point to build compact and full LLM context bundles.

    - Compact: README + example notebooks (converted to Markdown)
    - Full: Compact + all bayesflow/*.py files (excluding certain directories)
    """
    # Validate required inputs
    if not readme_file.exists():
        raise FileNotFoundError(f"README.md file not found: {readme_file}")
    if not examples_dir.exists():
        raise FileNotFoundError(f"examples directory not found: {examples_dir}")
    if not src_dir.exists():
        raise FileNotFoundError(f"bayesflow source directory not found: {src_dir}")

    # Prepare temporary examples directory under repo root so Repomix can use relative paths.
    with tempfile.TemporaryDirectory(prefix=".examples_temporary_", dir=str(base_dir)) as tmpdir:
        temp_examples_dir = Path(tmpdir)

        # Convert notebooks into the temp folder (no changes in the real examples/ directory)
        md_abs_paths = convert_notebooks_to_md_in_temp(examples_dir, temp_examples_dir)

        # Prefer relative paths for Repomix
        md_for_repomix = to_relative_paths(md_abs_paths, base_dir)

        # Include README if present (relative path so Repomix sees it correctly)
        if readme_file.exists():
            md_for_repomix.append(Path("README.md"))

        # ---- Compact: examples only ----
        run_repomix_with_file_list(
            md_for_repomix,
            compact_output_file,
            repo_cwd=base_dir
        )

        # ---- Full: examples + bayesflow .py files ----
        py_abs_paths = collect_py_abs_paths(src_dir)
        if not py_abs_paths:
            raise FileNotFoundError(f"No Python files found in bayesflow source directory: {src_dir}")
        py_for_repomix = to_relative_paths(py_abs_paths, base_dir)
        full_list = [*md_for_repomix, *py_for_repomix]
        run_repomix_with_file_list(
            full_list,
            full_output_file,
            repo_cwd=base_dir
        )

if __name__ == "__main__":
    main()