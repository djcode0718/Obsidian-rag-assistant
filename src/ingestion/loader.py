"""Loader for Obsidian Markdown vaults from directories and zip archives."""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, BinaryIO


@dataclass
class MarkdownNote:
    """Represents a single parsed Obsidian markdown note."""

    filename: str
    relative_path: str
    title: str
    raw_content: str
    cleaned_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_yaml_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Extracts YAML frontmatter (between --- delimiters) if present.

    Returns:
        tuple of (metadata_dict, content_without_frontmatter)
    """
    metadata: Dict[str, Any] = {}
    cleaned = content

    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        yaml_text = match.group(1)
        cleaned = content[match.end():]
        # Lightweight key-value YAML parser without hard pyyaml dependency
        for line in yaml_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                # Handle bracketed lists: [a, b, c]
                if val.startswith("[") and val.endswith("]"):
                    items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
                    metadata[key] = items
                else:
                    metadata[key] = val.strip("'\"")

    return metadata, cleaned


def _extract_title(filename: str, metadata: Dict[str, Any], content: str) -> str:
    """Derives a human-friendly title for the note.

    Precedence:
    1. 'title' in frontmatter metadata
    2. First top-level '# Heading' in markdown
    3. Filename without extension
    """
    if metadata.get("title"):
        return str(metadata["title"]).strip()

    # Search for first '# Title'
    h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    # Fallback to filename stem
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def load_notes_from_directory(directory_path: str | Path) -> List[MarkdownNote]:
    """Loads all markdown files from a local vault directory recursively.

    Args:
        directory_path: Absolute or relative path to vault folder.

    Returns:
        List of MarkdownNote instances.
    """
    base = Path(directory_path).resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Vault directory not found: {directory_path}")

    notes: List[MarkdownNote] = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in [".md", ".markdown"]:
            # Ignore hidden directories like .obsidian or .git
            parts = path.relative_to(base).parts
            if any(p.startswith(".") for p in parts):
                continue

            try:
                raw_text = path.read_text(encoding="utf-8", errors="replace")
                meta, clean = _extract_yaml_frontmatter(raw_text)
                title = _extract_title(path.name, meta, clean)
                rel_path = str(path.relative_to(base))

                notes.append(
                    MarkdownNote(
                        filename=path.name,
                        relative_path=rel_path,
                        title=title,
                        raw_content=raw_text,
                        cleaned_content=clean,
                        metadata=meta,
                    )
                )
            except Exception as e:
                # Skip unreadable file gracefully with warning
                print(f"[Warning] Failed to load {path}: {e}")

    return notes


def load_notes_from_zip(zip_source: str | Path | BinaryIO | bytes) -> List[MarkdownNote]:
    """Loads all markdown files from an uploaded zip archive safely in-memory.

    Args:
        zip_source: Path to zip, open file object, or raw bytes.

    Returns:
        List of MarkdownNote instances.
    """
    notes: List[MarkdownNote] = []

    if isinstance(zip_source, (str, Path)):
        zfile = zipfile.ZipFile(zip_source, "r")
    elif isinstance(zip_source, bytes):
        zfile = zipfile.ZipFile(io.BytesIO(zip_source), "r")
    else:
        zfile = zipfile.ZipFile(zip_source, "r")

    with zfile:
        for entry in zfile.infolist():
            if entry.is_dir():
                continue

            # Path traversal safety check
            norm_name = os.path.normpath(entry.filename)
            if norm_name.startswith("..") or norm_name.startswith("/") or norm_name.startswith("\\"):
                continue

            # Check markdown extension
            if norm_name.lower().endswith((".md", ".markdown")):
                # Filter out hidden files like __MACOSX or .obsidian
                parts = Path(norm_name).parts
                if any(p.startswith(".") or p.startswith("__") for p in parts):
                    continue

                try:
                    with zfile.open(entry) as f:
                        raw_bytes = f.read()
                        raw_text = raw_bytes.decode("utf-8", errors="replace")
                        meta, clean = _extract_yaml_frontmatter(raw_text)
                        fname = Path(norm_name).name
                        title = _extract_title(fname, meta, clean)

                        notes.append(
                            MarkdownNote(
                                filename=fname,
                                relative_path=norm_name,
                                title=title,
                                raw_content=raw_text,
                                cleaned_content=clean,
                                metadata=meta,
                            )
                        )
                except Exception as e:
                    print(f"[Warning] Failed reading zip entry {norm_name}: {e}")

    return notes
