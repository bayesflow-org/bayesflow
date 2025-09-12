#!/usr/bin/env python3
"""
Build two repomix LLM-context files, but write converted .md files into a temporary directory
so the real examples/ folder is never modified.

 - llm_context/llm_context_compact.md  -> examples only (from temp dir)
 - llm_context/llm_context_full.md     -> examples (temp dir) + bayesflow source code
"""
import json
import subprocess
from pathlib import Path
import tempfile
import sys
import shutil

base_dir = Path(__file__).parent.parent.resolve()
print("base_dir:", base_dir)

examples_dir = base_dir / "examples"
src_dir = base_dir / "bayesflow"
readme_file = base_dir / "README.md"
output_dir = base_dir / "llm_context"
compact_output_file = output_dir / "llm_context_compact.md"
full_output_file = output_dir / "llm_context_full.md"

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Safety checks
if not examples_dir.exists():
    print(f"ERROR: examples directory not found: {examples_dir}", file=sys.stderr)
    raise SystemExit(1)
if not src_dir.exists():
    print(f"WARNING: bayesflow source directory not found: {src_dir} -- full context will be skipped.", file=sys.stderr)

def convert_notebooks_to_md_in_temp(src_examples_dir: Path, temp_examples_dir: Path):
    """
    Convert .ipynb files to .md and write them into temp_examples_dir.
    Returns:
      - list of Path objects (absolute) to the markdown files created (for repomix input)
      - list of actual file paths created (for cleanup)
    """
    created_paths = []
    md_paths = []

    for ipynb_file in sorted(src_examples_dir.glob("*.ipynb")):
        with open(ipynb_file, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        parts = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "markdown":
                parts.append("".join(cell.get("source", [])))
            elif cell.get("cell_type") == "code":
                parts.append("```python\n" + "".join(cell.get("source", [])) + "\n```")

        # write into the temporary examples directory (never into the real examples/)
        md_file = temp_examples_dir / f"{ipynb_file.stem}.md"

        # ensure unique name just in case (temp dir typically empty, but keep behaviour consistent)
        if md_file.exists():
            i = 1
            while True:
                candidate = temp_examples_dir / f"{ipynb_file.stem}.repomix.{i}.md"
                if not candidate.exists():
                    md_file = candidate
                    break
                i += 1

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts))

        created_paths.append(md_file)
        md_paths.append(md_file.resolve())

        print("Created temporary md:", md_file)

    return md_paths, created_paths

def collect_bayesflow_py_abs_paths(src_bayesflow_dir: Path):
    """Return a sorted list of absolute Paths for all .py files in src_bayesflow_dir."""
    return sorted(p.resolve() for p in sorted(src_bayesflow_dir.rglob("*.py")))

def run_repomix_with_file_list(file_paths, output_path, repo_cwd, include_patterns="**/*.py,**/*.md"):
    """Run repomix (cwd=repo_cwd) with --stdin reading newline-separated paths (absolute or relative)."""
    if not file_paths:
        print(f"No files provided for repomix output {output_path}. Skipping.", file=sys.stderr)
        return

    cmd = [
        "repomix",
        "--style", "markdown",
        "--stdin",
        "--include", include_patterns,
        "--ignore", "bayesflow/experimental/",
        "-o", str(output_path),
    ]
    print(f"Running repomix in cwd={repo_cwd}: {' '.join(cmd)}")
    print(f"  -> {len(file_paths)} files (showing up to 20):")
    for p in file_paths[:20]:
        print("    ", str(p))

    stdin_input = "\n".join(str(p) for p in file_paths) + "\n"
    subprocess.run(cmd, input=stdin_input, text=True, check=True, cwd=str(repo_cwd))
    print(f"✅ Repomix packaged output saved to {output_path}")

# --- Main flow ---
# Create a temporary examples directory *under the repo root* so repomix can use relative paths if it wants.
temp_dir_path = None
created_files = []

try:
    temp_dir = tempfile.mkdtemp(prefix=".examples_temporary_", dir=str(base_dir))
    temp_examples_dir = Path(temp_dir)
    temp_dir_path = temp_examples_dir
    print("Using temporary examples dir:", temp_examples_dir)

    # Convert notebooks into the temp folder (no changes in the real examples/ directory)
    md_abs_paths, created_files = convert_notebooks_to_md_in_temp(examples_dir, temp_examples_dir)
    if not md_abs_paths:
        print("ERROR: No example notebooks (*.ipynb) found or conversion produced no markdown files.", file=sys.stderr)
        raise SystemExit(1)

    # For repomix we can pass relative paths (relative to repo root) — convert if possible
    try:
        md_rel_for_repomix = [p.relative_to(base_dir) for p in md_abs_paths]
    except Exception:
        # fallback to absolute paths if relative conversion fails
        md_rel_for_repomix = md_abs_paths

    # Include README if present (use relative path so repomix sees it correctly)
    if readme_file.exists():
        print("Including top-level README.md in repomix inputs")
        md_rel_for_repomix.append(Path("README.md"))

    # ---- Compact: examples only ----
    run_repomix_with_file_list(md_rel_for_repomix, compact_output_file, repo_cwd=base_dir, include_patterns="**/*.md")

    # ---- Full: examples + bayesflow .py files ----
    if src_dir.exists():
        py_abs_paths = collect_bayesflow_py_abs_paths(src_dir)
        # convert py paths to relative if possible
        try:
            py_rel_for_repomix = [p.relative_to(base_dir) for p in py_abs_paths]
        except Exception:
            py_rel_for_repomix = py_abs_paths

        full_list = md_rel_for_repomix + py_rel_for_repomix
        run_repomix_with_file_list(full_list, full_output_file, repo_cwd=base_dir, include_patterns="**/*.py,**/*.md")
    else:
        print("Skipping creation of full context because bayesflow directory was not found.", file=sys.stderr)

finally:
    # Clean up only the temporary files / dir we created
    if created_files:
        for p in created_files:
            try:
                if p.exists():
                    p.unlink()
                    print("Removed temporary md:", p)
            except Exception as e:
                print(f"Warning: failed to remove {p}: {e}", file=sys.stderr)

    if temp_dir_path and temp_dir_path.exists():
        try:
            shutil.rmtree(temp_dir_path)
            print("Removed temporary directory:", temp_dir_path)
        except Exception as e:
            print(f"Warning: failed to remove temporary directory {temp_dir_path}: {e}", file=sys.stderr)

print("Done.")
