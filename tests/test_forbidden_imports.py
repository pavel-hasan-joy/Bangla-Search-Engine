"""Automated verification test scanning all core source files for forbidden third-party libraries.

Per repository rules:
Core engine must be hand-written with the Python standard library only.
Forbidden in core:
- Lucene, Elasticsearch, Whoosh, Tantivy
- rank_bm25
- scikit-learn, sklearn
- nltk, spacy, transformers, torch, gensim, fasttext, polyglot
- indic_transliteration, pyavro, googletrans, sentence_transformers
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_PACKAGES = frozenset([
    "lucene",
    "elasticsearch",
    "elastic",
    "whoosh",
    "tantivy",
    "rank_bm25",
    "sklearn",
    "scikit_learn",
    "scikitlearn",
    "nltk",
    "spacy",
    "transformers",
    "torch",
    "gensim",
    "fasttext",
    "polyglot",
    "indic_transliteration",
    "pyavro",
    "googletrans",
    "sentence_transformers",
    "langchain",
    "llama_index",
])

# Root path for core source packages
SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "bsearch"


def extract_imports(file_path: Path) -> set[str]:
    """Parse python file with AST and extract all top-level imported package names."""
    imports = set()
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except Exception as e:
        raise RuntimeError(f"Failed to parse {file_path}: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0].lower()
                imports.add(root_pkg)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0].lower()
                imports.add(root_pkg)

    return imports


def test_no_forbidden_imports_in_core():
    """Scan all Python files under src/bsearch/ to ensure zero forbidden libraries are used."""
    assert SRC_DIR.is_dir(), f"Source directory not found: {SRC_DIR}"

    py_files = list(SRC_DIR.rglob("*.py"))
    assert len(py_files) > 0, "No Python source files found to scan"

    violations: list[str] = []

    for py_file in py_files:
        imported_pkgs = extract_imports(py_file)
        forbidden_used = imported_pkgs.intersection(FORBIDDEN_PACKAGES)
        if forbidden_used:
            violations.append(
                f"{py_file.relative_to(SRC_DIR.parent)} imports forbidden packages: {sorted(forbidden_used)}"
            )

    assert not violations, "\n".join(["Forbidden library imports detected:"] + violations)
