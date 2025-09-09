"""
Build BayesFlow LLM context files (full + compact).

Artifacts written to llm_context/:
- bayesflow-context-full-<TAG>.md
- bayesflow-context-compact-<TAG>.md
- bayesflow-context-<TAG>.manifest.json

Strategy:
- Convert notebooks in examples/ to Markdown (temporary, not committed).
- Run repomix on bayesflow/, examples/, README.md.
- Compact file: README + examples fully, bayesflow/ truncated unless short.
- Both files include a short dependency summary from pyproject.toml.
"""

from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
import re
import nbformat
from typing import List, Optional, Tuple

# Configuration
ROOT = Path(".").resolve()
OUT_DIR = Path("llm_context")
INCLUDE_FOLDERS = ("bayesflow/",)
INCLUDE_FILES = ("README.md",)
PYPROJECT = Path("pyproject.toml")
HEADING_RE = re.compile(r"^\s#{2,}\s*(?:FILE:\s*)?(?P<path>.+?)\s*$", flags=re.MULTILINE)
TOKEN_CHAR_RATIO = 4

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Utilities
def run(cmd: List[str], input_text: Optional[str] = None) -> str:
    """
    Run a shell command and capture stdout.

    Parameters
    ----------
    cmd : list of str
        Command and arguments.
    input_text : str, optional
        Text passed to stdin.

    Returns
    -------
    str
        Captured stdout.

    Raises
    ------
    RuntimeError
        If the command exits with a non-zero status.
    """
    res = subprocess.run(cmd, check=False, text=True, input=input_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr.strip()}")
    return res.stdout


def token_estimate(text: str) -> int:
    """
    Roughly estimate token count for text.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Estimated token count.
    """
    return max(1, len(text) // TOKEN_CHAR_RATIO)


def load_dependency_summary(pyproject: Path = PYPROJECT) -> List[str]:
    """
    Extract dependencies from pyproject.toml.

    Parameters
    ----------
    pyproject : Path
        Path to pyproject.toml.

    Returns
    -------
    list of str
        Dependency strings, or empty list if not available.
    """
    if not pyproject.exists():
        return []
    try:
        import tomllib as _toml  # Python 3.11+
    except Exception:
        import tomli as _toml  # Fallback
    try:
        data = _toml.loads(pyproject.read_text(encoding="utf8"))
    except Exception:
        return []
    proj = data.get("project", {}) or {}
    raw = proj.get("dependencies", []) or []
    return [d.split(";")[0].strip() for d in raw if isinstance(d, str)]


# Notebook conversion
def notebook_to_md(nb_path: Path) -> str:
    """
    Convert Jupyter notebook to Markdown.

    Parameters
    ----------
    nb_path : Path
        Path to .ipynb file.

    Returns
    -------
    str
        Markdown text with markdown and code cells.
    """
    nb = nbformat.read(str(nb_path), as_version=4)
    out: List[str] = [f"# Notebook: {nb_path.name}", ""]
    for cell in nb.cells:
        src = "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list) else cell.get("source", "")
        if cell.get("cell_type") == "markdown":
            out.append(src.rstrip())
            out.append("")
        elif cell.get("cell_type") == "code" and src.strip():
            out.extend(["```python", src.strip("\n"), "```", ""])
    return "\n".join(out).rstrip() + "\n"


def convert_examples_to_md(src: Path, out: Path) -> List[Path]:
    """
    Convert all .ipynb notebooks in a directory tree to Markdown.

    Parameters
    ----------
    src : Path
        Source examples/ directory.
    out : Path
        Destination directory for converted .md files.

    Returns
    -------
    list of Path
        List of generated Markdown file paths.
    """
    created: List[Path] = []
    if not src.exists():
        return created
    out.mkdir(parents=True, exist_ok=True)
    for nb in sorted(src.rglob("*.ipynb")):
        try:
            dst = out / (nb.stem + ".md")
            dst.write_text(notebook_to_md(nb), encoding="utf8")
            created.append(dst)
            logging.info("Converted %s -> %s", nb, dst)
        except Exception as e:
            logging.warning("Failed to convert %s: %s", nb, e)
    return created


# Context generation
def run_repomix_on_paths(paths: List[str], style: str = "markdown") -> str:
    """
    Run repomix on given paths.

    Parameters
    ----------
    paths : list of str
        Relative paths to include.
    style : str
        Output style, default 'markdown'.

    Returns
    -------
    str
        Repomix output.
    """
    cmd = ["repomix", "--style", style, "--stdin", "--stdout"]
    return run(cmd, input_text="\n".join(paths) + "\n")


def generate_compact(full_text: str, tag: str, repo_root: Path, conv_examples_dir: Path) -> str:
    """
    Create compact context file from full repomix output.

    Parameters
    ----------
    full_text : str
        Full repomix output.
    tag : str
        Release tag.
    repo_root : Path
        Repository root.
    conv_examples_dir : Path
        Path to temporary converted examples.

    Returns
    -------
    str
        Compact context content.
    """
    lines = full_text.splitlines(keepends=True)
    sections: List[Tuple[str, int, int]] = []
    cur_path: Optional[str] = None
    cur_start = 0
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            if cur_path is not None:
                sections.append((cur_path, cur_start, i))
            cur_path = m.group("path").strip()
            cur_start = i + 1
    if cur_path is not None:
        sections.append((cur_path, cur_start, len(lines)))

    if not sections:
        return f"<!-- Compact artifact for BayesFlow {tag} (fallback) -->\n\n{''.join(lines[:40])}"

    out_lines: List[str] = [f"<!-- Compact artifact for BayesFlow {tag} -->\n\n"]
    preview_lines = 40
    max_keep_tokens = 1200
    for idx, (path, s, e) in enumerate(sections, start=1):
        seg = "".join(lines[s:e])
        path_lower = path.lower()
        keep_full = (
            path_lower.endswith("readme.md")
            or path_lower.startswith("examples")
            or (conv_examples_dir.exists() and (conv_examples_dir / Path(path).name).exists())
            or (path.startswith("bayesflow") and token_estimate(seg) <= max_keep_tokens)
        )
        out_lines.append(f"## {path}  <!-- chunk {idx} lines {s + 1}-{e} -->\n\n")
        if keep_full:
            out_lines.append(seg + "\n")
        else:
            out_lines.append("".join(seg.splitlines(keepends=True)[:preview_lines]))
            out_lines.append(f"\n> [TRUNCATED] See full file for `{path}` lines {s + 1}-{e}.\n\n")
    return "".join(out_lines)


# Build pipeline
def build(tag: Optional[str], out_dir: Path):
    """
    Generate full + compact context files and manifest.

    Parameters
    ----------
    tag : str or None
        Release tag. If None, inferred from environment or commit hash.
    out_dir : Path
        Destination directory.

    Returns
    -------
    tuple of Path
        (full_file, compact_file, manifest_file)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    deps = load_dependency_summary(PYPROJECT)
    dep_md = "**Dependency summary:**\n" + "\n".join(f"- {d}" for d in deps) + "\n\n" if deps else ""

    with tempfile.TemporaryDirectory(prefix="bf-conv-") as tmp:
        tmp_path = Path(tmp)
        convert_examples_to_md(ROOT / "examples", tmp_path)
        repomix_inputs = [str(p) for p in INCLUDE_FOLDERS if (ROOT / p).exists()]
        if tmp_path.exists():
            repomix_inputs.append(str(tmp_path))
        for f in INCLUDE_FILES:
            if (ROOT / f).exists():
                repomix_inputs.append(f)
        repomix_out = run_repomix_on_paths(repomix_inputs, style="markdown")

    try:
        commit = run(["git", "rev-parse", "HEAD"]).strip()
    except Exception:
        commit = None

    tag = (
        tag
        or os.environ.get("RELEASE_TAG")
        or (commit[:7] if commit else datetime.datetime.utcnow().strftime("%Y%m%d"))
    )
    header = {
        "artifact": f"bayesflow-context-full-{tag}.md",
        "tag": tag,
        "commit": commit,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    header_block = ["---"] + [f"{k}: {v}" for k, v in header.items() if v] + ["---", ""]

    full_text = "\n".join(header_block) + dep_md + repomix_out
    full_path = out_dir / f"bayesflow-context-full-{tag}.md"
    full_path.write_text(full_text, encoding="utf8")

    compact_text = generate_compact(repomix_out, tag, ROOT, tmp_path)
    compact_path = out_dir / f"bayesflow-context-compact-{tag}.md"
    compact_path.write_text("\n".join(header_block) + dep_md + compact_text, encoding="utf8")

    manifest = {
        "tag": tag,
        "commit": commit,
        "generated_at": header["generated_at"],
        "dependency_summary": deps,
        "files": {
            full_path.name: {"size_bytes": full_path.stat().st_size},
            compact_path.name: {"size_bytes": compact_path.stat().st_size},
        },
    }
    manifest_path = out_dir / f"bayesflow-context-{tag}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")

    logging.info("Built artifacts: %s, %s, %s", full_path, compact_path, manifest_path)
    return full_path, compact_path, manifest_path


def main(argv=None):
    """
    CLI entrypoint.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    int
        Exit status code.
    """
    parser = argparse.ArgumentParser(description="Build BayesFlow LLM context (full + compact).")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args(argv)
    build(args.tag, OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
