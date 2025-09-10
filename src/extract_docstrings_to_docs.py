# src/extract_docstrings_to_docs.py
"""
Extract module-level docstrings from Python files in src/ and write each to docs/<module>.md.
Also creates docs/README.md as an index.
Usage:
    python src/extract_docstrings_to_docs.py
"""
import ast
from pathlib import Path
import textwrap

SRC_DIR = Path("src")
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

def extract_docstring(py_path: Path) -> str:
    try:
        source = py_path.read_text(encoding="utf-8")
        module = ast.parse(source)
        doc = ast.get_docstring(module)
        return doc or ""
    except Exception as e:
        return f"**Error reading docstring:** {e}"

def safe_md_name(py_path: Path) -> Path:
    name = py_path.stem
    # replace problematic chars with underscore
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
    return DOCS_DIR / f"{safe}.md"

def write_md(py_path: Path):
    doc = extract_docstring(py_path)
    md_path = safe_md_name(py_path)
    title = py_path.name
    header = f"# {title}\n\n"
    if doc:
        body = textwrap.dedent(doc).strip() + "\n"
        content = header + body
    else:
        content = header + "_No module-level docstring found. See source file for inline comments._\n"
    md_path.write_text(content, encoding="utf-8")
    return md_path

def make_index(md_paths):
    index_path = DOCS_DIR / "README.md"
    lines = [
        "# Documentation index\n\n",
        "This folder contains extracted module-level docstrings from `src/`.\n\n",
        "## Modules\n\n"
    ]
    for p in sorted(md_paths, key=lambda x: x.name.lower()):
        summary = ""
        try:
            text = p.read_text(encoding="utf-8")
            # pick first non-empty line after the title
            for ln in text.splitlines()[1:]:
                ln = ln.strip()
                if ln:
                    summary = ln
                    break
        except Exception:
            summary = ""
        lines.append(f"- [{p.stem}]({p.name}) — {summary}\n")
    index_path.write_text("".join(lines), encoding="utf-8")
    return index_path

def main():
    if not SRC_DIR.exists():
        print("src/ directory not found. Please run this script from repository root.")
        return
    py_files = [p for p in sorted(SRC_DIR.glob("*.py")) if p.name != Path(__file__).name]
    if not py_files:
        print("No Python files found in src/.")
        return
    md_paths = []
    for py in py_files:
        md = write_md(py)
        md_paths.append(md)
        print(f"Written {md}")
    idx = make_index(md_paths)
    print(f"Index written: {idx}")

if __name__ == "__main__":
    main()
