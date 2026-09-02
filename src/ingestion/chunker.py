"""Header-aware Markdown chunking engine with metadata preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None


def count_tokens(text: str) -> int:
    """Estimates or calculates exact token count for a text snippet."""
    if _TIKTOKEN_ENCODER is not None:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text))
        except Exception:
            pass
    # Reliable rule-of-thumb: ~0.75 words per token (or ~1.3 tokens per word)
    words = len(text.split())
    return max(1, int(words * 1.3))


@dataclass
class MarkdownChunk:
    """Represents a discrete semantic chunk of a markdown note."""

    chunk_id: str
    text: str
    source_file: str
    relative_path: str
    note_title: str
    heading: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def _split_into_sections(content: str) -> List[tuple[str, str]]:
    """Splits markdown content by headings (#, ##, ###, ####).

    Returns:
        List of tuples: (heading_text, section_body)
    """
    lines = content.splitlines()
    sections: List[tuple[str, str]] = []

    current_heading = "Introduction"
    current_lines: List[str] = []

    heading_regex = re.compile(r"^(#{1,4})\s+(.+)$")

    for line in lines:
        match = heading_regex.match(line)
        if match:
            # Commit previous section if non-empty
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((current_heading, section_text))
                current_lines = []
            current_heading = match.group(2).strip()
        else:
            current_lines.append(line)

    final_text = "\n".join(current_lines).strip()
    if final_text:
        sections.append((current_heading, final_text))

    return sections


def _split_text_with_overlap(
    text: str,
    target_tokens: int = 500,
    overlap_tokens: int = 50
) -> List[str]:
    """Splits a single text string into overlapping token windows."""
    total_tokens = count_tokens(text)
    if total_tokens <= target_tokens:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current_paras: List[str] = []
    current_count = 0

    for p in paragraphs:
        p_tokens = count_tokens(p)
        if current_count + p_tokens > target_tokens and current_paras:
            # Flush current window
            chunk_str = "\n\n".join(current_paras).strip()
            chunks.append(chunk_str)

            # Keep last paragraph for overlap if it doesn't exceed overlap budget
            overlap_accum: List[str] = []
            overlap_count = 0
            for prev_p in reversed(current_paras):
                cnt = count_tokens(prev_p)
                if overlap_count + cnt <= overlap_tokens:
                    overlap_accum.insert(0, prev_p)
                    overlap_count += cnt
                else:
                    break

            current_paras = overlap_accum
            current_count = overlap_count

        current_paras.append(p)
        current_count += p_tokens

    if current_paras:
        final_chunk = "\n\n".join(current_paras).strip()
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks


def chunk_notes(
    notes: List[Any],
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 50,
) -> List[MarkdownChunk]:
    """Transforms a collection of MarkdownNote objects into indexed MarkdownChunk objects.

    Args:
        notes: List of MarkdownNote instances.
        chunk_size_tokens: Target token count per chunk.
        chunk_overlap_tokens: Overlap tokens between sequential chunks.

    Returns:
        List of MarkdownChunk instances ready for vector embedding.
    """
    all_chunks: List[MarkdownChunk] = []

    for note in notes:
        content = note.cleaned_content or note.raw_content
        sections = _split_into_sections(content)

        chunk_idx = 0
        for heading, section_text in sections:
            # Decompose section if larger than target chunk size
            window_chunks = _split_text_with_overlap(
                section_text,
                target_tokens=chunk_size_tokens,
                overlap_tokens=chunk_overlap_tokens
            )

            for w_text in window_chunks:
                # Prepend contextual header banner for rich semantic grounding
                formatted_text = f"## [{note.title}] > {heading}\n{w_text}"
                n_tokens = count_tokens(formatted_text)
                chunk_id = f"{note.filename}__s{chunk_idx}"

                meta = {
                    "source_file": note.filename,
                    "relative_path": note.relative_path,
                    "note_title": note.title,
                    "heading": heading,
                    "chunk_index": chunk_idx,
                    "token_count": n_tokens,
                }
                # Include any frontmatter tags if available
                if "tags" in note.metadata:
                    meta["tags"] = str(note.metadata["tags"])

                all_chunks.append(
                    MarkdownChunk(
                        chunk_id=chunk_id,
                        text=formatted_text,
                        source_file=note.filename,
                        relative_path=note.relative_path,
                        note_title=note.title,
                        heading=heading,
                        chunk_index=chunk_idx,
                        token_count=n_tokens,
                        metadata=meta,
                    )
                )
                chunk_idx += 1

    return all_chunks
